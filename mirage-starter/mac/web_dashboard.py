#!/usr/bin/env python3
"""web_dashboard.py: the Mirage console in a browser.                         owner: Gio

Shows the same data as dashboard.py, read the same way: the supervisor's telemetry JSON on
UDP 5005 and the vision process's JPEG shift-map thumbnails on UDP 5006. While the Pi is live,
the image panel shows mirage_vision's own camera view of the dot screen region instead
(http://<pi>:8080/region). Its "New reference frame" and "Open valve" buttons are the only things
the console sends the Pi, through mirage_vision; the supervisor still decides whether the valve
opens. They work from this computer, or from devices named with --allow-control.
While it runs it owns UDP 5005 and 5006, so dashboard.py
(still the fallback) can't run at the same time, and the other way round.

  python mac/web_dashboard.py                  serves http://127.0.0.1:8750 and opens it
  python mac/web_dashboard.py --no-browser
  python mac/web_dashboard.py --host 0.0.0.0   also reachable from another screen on the network

The page can start and stop fake_pi.py for rehearsals, only from this Mac and never while
real telemetry is arriving. test_web_dashboard.py checks it against dashboard.py itself.
"""
import argparse
import collections
import json
import math
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

MAC_DIR = Path(__file__).resolve().parent
WEB_DIR = MAC_DIR / "web"
MSGS_H = MAC_DIR.parent / "common" / "mirage_msgs.h"

# Read exactly like dashboard.py: same ports, buffer sizes, labels, stale rule and history lengths.
TEL_PORT, IMG_PORT = 5005, 5006
TEL_BUF, IMG_BUF = 2048, 65535
LABELS = ["clear", "gas", "heat", "motion"]
STALE_S = 1.0                 # no telemetry for longer than this: "no telemetry", valve not shown open
HISTORY_LEN = 120             # (p_gas, tripped) per telemetry packet
LAT_LEN = 600                 # lat_ms values above 0, for the p99

SOURCE_ACTIVE_S = STALE_S     # a sender counts as active if heard from within the stale window
LIVE_GUARD_S = 5.0            # the simulator won't start within this long of real telemetry
STREAM_PERIOD_S = 0.1         # each browser gets a snapshot 10 times a second
MAX_STREAMS = 16
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
PI_VIEW_PORT = 8080                                  # mirage_vision -p: its browser view on the Pi


def finite(obj):
    """JSON-safe copy: NaN and infinities become null (the browser's JSON parser rejects them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def contract_constants():
    """Trip rules from common/mirage_msgs.h, for labels only. The supervisor enforces them."""
    found = {"deadline_ms": 250, "trip_run": 3, "reset_clear": 5, "p_gas_trip": 0.60}
    names = {"deadline_ms": "MIRAGE_DEADLINE_MS", "trip_run": "MIRAGE_TRIP_RUN",
             "reset_clear": "MIRAGE_RESET_CLEAR", "p_gas_trip": "MIRAGE_P_GAS_TRIP"}
    source = "built-in defaults"
    try:
        text = MSGS_H.read_text()
        for key, name in names.items():
            m = re.search(rf"#define\s+{name}\s+([0-9.]+)", text)
            if m:
                found[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        source = "common/mirage_msgs.h"
    except OSError:
        pass
    return {**found, "stale_s": STALE_S, "history_len": HISTORY_LEN, "source": source}


def local_addresses():
    """This Mac's LAN address, to recognize packets that come from here (fake_pi.py)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))         # TEST-NET address; connect() on UDP sends nothing
            return [s.getsockname()[0]]
    except OSError:
        return []


class Console:
    """Everything the page shows. Fed by the UDP threads, read by the HTTP threads."""

    def __init__(self, host_ips=()):
        self.lock = threading.Lock()
        self.host_ips = list(host_ips)
        self.local_ips = LOOPBACK | set(host_ips)
        self.constants = contract_constants()
        self.state, self.last_rx = {}, None
        self.history = collections.deque(maxlen=HISTORY_LEN)
        self.lat = collections.deque(maxlen=LAT_LEN)
        self.packets = self.malformed = 0
        self.rx_times = collections.deque(maxlen=64)
        self.sources = {}                       # (ip, port) -> [last_rx, packets]
        self.latest_source = None
        self.thumb, self.thumb_seq, self.thumb_rejected = None, 0, 0
        self.thumb_rx = self.thumb_shape = self.thumb_source = None

    def on_telemetry(self, data, addr, now):
        """One datagram from UDP 5005, interpreted exactly as dashboard.py does."""
        try:
            state = json.loads(data.decode())
        except ValueError:                      # a stray or truncated packet
            with self.lock:
                self.malformed += 1
            return False
        if not isinstance(state, dict):         # valid JSON but not an object: dashboard.py raises here
            with self.lock:
                self.malformed += 1
            return False
        lat = state.get("lat_ms", 0)
        with self.lock:
            self.state, self.last_rx = state, now
            self.history.append((state.get("p_gas", 0.0), state.get("tripped", 1), state.get("t_ms")))
            if isinstance(lat, (int, float)) and lat > 0:
                self.lat.append(lat)
            self.packets += 1
            self.rx_times.append(now)
            src = (addr[0], addr[1])
            entry = self.sources.setdefault(src, [now, 0])
            entry[0], entry[1] = now, entry[1] + 1
            self.latest_source = src
            for key in [k for k, (t, _) in self.sources.items() if now - t > 60.0]:
                del self.sources[key]           # keep the table small
        return True

    def on_image(self, data, addr, now):
        """One datagram from UDP 5006: kept only if it decodes, as in dashboard.py."""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        with self.lock:
            if img is None:
                self.thumb_rejected += 1
                return False
            self.thumb, self.thumb_seq = bytes(data), self.thumb_seq + 1
            self.thumb_rx, self.thumb_shape, self.thumb_source = now, img.shape[:2], addr[0]
        return True

    def thumbnail(self):
        with self.lock:
            return self.thumb, self.thumb_seq

    def active_sources(self, now, within):
        return [src for src, (t, _) in self.sources.items() if now - t <= within]

    def sim_block_reason(self, now, sim_running):
        """Why the simulator must not start now, or None."""
        if sim_running:
            return None
        with self.lock:
            live = [ip for ip, _ in self.active_sources(now, LIVE_GUARD_S) if ip not in self.local_ips]
            local = [ip for ip, _ in self.active_sources(now, SOURCE_ACTIVE_S) if ip in self.local_ips]
        if live:
            return f"Live telemetry from {live[0]}. The simulator would mix with it."
        if local:
            return "Something on this Mac is already sending telemetry (another fake_pi.py?)."
        return None

    def snapshot(self, now):
        """What dashboard.py would draw at time `now`, plus the extra context the page shows."""
        with self.lock:
            state = dict(self.state)
            stale = self.last_rx is None or now - self.last_rx > STALE_S
            try:
                class_label = LABELS[state["cls"]] if "cls" in state and state["cls"] < 4 else "?"
            except (TypeError, IndexError):
                class_label = "?"
            sources = sorted(({"addr": ip, "port": port, "local": ip in self.local_ips,
                               "age_s": now - t, "packets": n, "active": now - t <= SOURCE_ACTIVE_S}
                              for (ip, port), (t, n) in self.sources.items()), key=lambda s: s["age_s"])
            latest_local = self.latest_source is not None and self.latest_source[0] in self.local_ips
            tel = {
                "received": self.packets > 0,
                "age_s": None if self.last_rx is None else now - self.last_rx,
                "stale": stale,
                "valve_open": state.get("valve") == "open" and not stale,
                "class_label": class_label,
                "state": state,
                "history": [list(h) for h in self.history],
                "lat_p99": float(np.percentile(np.asarray(self.lat), 99)) if self.lat else None,
                "lat_count": len(self.lat),
                "packets": self.packets,
                "malformed": self.malformed,
                "rate_hz": sum(1 for t in self.rx_times if now - t <= 2.0) / 2.0,
                "sources": sources,
                "multi_source": sum(1 for s in sources if s["active"]) > 1,
                "latest_source": None if self.latest_source is None else self.latest_source[0],
                "simulated": latest_local,
            }
            img = {
                "seq": self.thumb_seq,
                "age_s": None if self.thumb_rx is None else now - self.thumb_rx,
                "width": None if self.thumb_shape is None else int(self.thumb_shape[1]),
                "height": None if self.thumb_shape is None else int(self.thumb_shape[0]),
                "bytes": 0 if self.thumb is None else len(self.thumb),
                "source": self.thumb_source,
                "simulated": self.thumb_source in self.local_ips if self.thumb_source else False,
                "rejected": self.thumb_rejected,
            }
        return finite({"telemetry": tel, "image": img, "constants": self.constants,
                       "host_ips": self.host_ips, "ports": {"telemetry": TEL_PORT, "image": IMG_PORT}})


class Simulator:
    """fake_pi.py as a child process, started and stopped from the page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def pid(self):
        return self.proc.pid if self.running() else None

    def start(self):
        with self.lock:
            if not self.running():
                self.proc = subprocess.Popen([sys.executable, str(MAC_DIR / "fake_pi.py"), "--ip", "127.0.0.1"],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        with self.lock:
            if self.running():
                self.proc.terminate()
                try:
                    self.proc.wait(2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(2)
            self.proc = None


STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8")}


def make_server(console, sim, host, port, stop_event, pi_view_port=PI_VIEW_PORT, allow_control=()):
    streams = threading.BoundedSemaphore(MAX_STREAMS)
    allow_control = set(allow_control)

    def pi_address(snap):
        """The Pi sending live telemetry, or None: nothing yet, or it comes from this Mac."""
        t = snap["telemetry"]
        return t["latest_source"] if t["received"] and not t["simulated"] else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "MirageConsole/1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _snapshot(self):
            now = time.monotonic()
            snap = console.snapshot(now)
            snap["sim"] = {"running": sim.running(), "pid": sim.pid(),
                           "blocked": console.sim_block_reason(now, sim.running())}
            # mirage_vision's own web view on the Pi: the page shows its /region stream
            snap["pi_view"] = {"host": pi_address(snap), "port": pi_view_port}
            snap["can_control"] = self._pi_control_allowed(require_header=False)   # for this viewer
            return json.dumps(snap, allow_nan=False).encode()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in STATIC:
                name, ctype = STATIC[path]
                try:
                    self._send(200, (WEB_DIR / name).read_bytes(), ctype)
                except OSError:
                    self._send(404, b"missing web file\n", "text/plain")
            elif path == "/api/state":
                self._send(200, self._snapshot(), "application/json")
            elif path == "/api/thumb.jpg":
                jpg, _ = console.thumbnail()
                if jpg is None:
                    self._send(404, b"no image yet\n", "text/plain")
                else:
                    self._send(200, jpg, "image/jpeg")
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(404, b"not found\n", "text/plain")

        def _stream(self):
            if not streams.acquire(blocking=False):
                self._send(503, b"too many open pages\n", "text/plain")
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.connection.settimeout(5.0)     # a page that stops reading gets dropped
                while not stop_event.is_set():
                    self.wfile.write(b"data: " + self._snapshot() + b"\n\n")
                    self.wfile.flush()
                    stop_event.wait(STREAM_PERIOD_S)
            except OSError:
                pass                                # the page closed or went away
            finally:
                streams.release()

        def _control_allowed(self):
            """Only a page served to this Mac, not another site or another machine."""
            if self.client_address[0] not in LOOPBACK or self.headers.get("X-Mirage-Control") != "1":
                return False
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != host:
                return False
            return host.rsplit(":", 1)[0] in ("127.0.0.1", "localhost", "[::1]")

        def _pi_control_allowed(self, require_header=True):
            """New reference frame and Open valve: this computer, by any of its addresses, or a
            device named with --allow-control, and only from this console's own page."""
            client = self.client_address[0]
            if client not in console.local_ips and client not in allow_control:
                return False
            if require_header and self.headers.get("X-Mirage-Control") != "1":
                return False
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != host:
                return False
            hostname = host.rsplit(":", 1)[0]
            return hostname in ("localhost", "[::1]") or hostname in console.local_ips

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(min(length, 4096))
            path = urlsplit(self.path).path
            if path == "/api/reference":
                self._new_reference()
                return
            if path == "/api/reset":
                self._reset_valve()
                return
            if path not in ("/api/sim/start", "/api/sim/stop"):
                self._json(404, {"error": "not found"})
                return
            if not self._control_allowed():
                self._json(403, {"error": "The simulator can only be controlled from this Mac."})
                return
            if path == "/api/sim/start":
                reason = console.sim_block_reason(time.monotonic(), sim.running())
                if reason:
                    self._json(409, {"error": reason})
                    return
                try:
                    sim.start()
                except OSError as e:
                    self._json(500, {"error": f"Couldn't start fake_pi.py: {e}"})
                    return
            else:
                sim.stop()
            self._json(200, {"running": sim.running()})

        def _post_to_pi(self, what, path):
            """Forwards a button press to mirage_vision's web view on the Pi. Returns (pi, HTTP status,
            body), or None after answering the page itself (refused, no Pi, or no answer)."""
            if not self._pi_control_allowed():
                self._json(403, {"error": f"{what} only works from this computer or an --allow-control device."})
                return None
            pi = pi_address(console.snapshot(time.monotonic()))
            if not pi:
                self._json(409, {"error": "No live telemetry from the Pi, so there's no Pi to ask."})
                return None
            req = urllib.request.Request(f"http://{pi}:{pi_view_port}{path}", data=b"", method="POST")
            try:
                with urllib.request.urlopen(req, timeout=3) as r:
                    return pi, r.status, r.read().decode(errors="replace").strip()
            except urllib.error.HTTPError as e:            # the Pi answered, with a refusal
                return pi, e.code, e.read().decode(errors="replace").strip()
            except OSError as e:
                self._json(502, {"error": "mirage_vision on the Pi didn't answer."})
                return None

        def _new_reference(self):
            """Ask mirage_vision for a new reference frame."""
            answer = self._post_to_pi("A new reference frame", "/reference")
            if answer:
                pi, status, _ = answer
                if status == 200:
                    self._json(200, {"ok": True})
                else:
                    self._json(502, {"error": f"mirage_vision on the Pi refused ({status})."})

        def _reset_valve(self):
            """Ask the supervisor, through mirage_vision, to reopen the valve. Its rules decide."""
            answer = self._post_to_pi("Opening the valve", "/reset")
            if answer:
                pi, status, body = answer
                if body == "accepted":
                    self._json(200, {"result": "accepted"})
                elif body == "refused":
                    self._json(409, {"result": "refused",
                                     "error": "The supervisor refused: gas is still in view, or vision isn't sending."})
                else:
                    self._json(502, {"result": body or str(status),
                                     "error": "The supervisor on the Pi isn't reachable from mirage_vision."})

    return ThreadingHTTPServer((host, port), Handler)


def bind_udp(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))               # as dashboard.py: no SO_REUSEPORT, so nobody steals packets
    except OSError as e:
        s.close()
        raise SystemExit(f"UDP {port} is already in use ({e.strerror}). "
                         "Quit dashboard.py or the other console first.")
    s.settimeout(0.5)
    return s


def receive(sock, bufsize, handler, stop_event):
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(bufsize)
        except socket.timeout:
            continue
        except OSError:
            if stop_event.is_set():
                break
            continue
        handler(data, addr, time.monotonic())


def main():
    ap = argparse.ArgumentParser(description="Mirage console in a browser (same data as dashboard.py)")
    ap.add_argument("--host", default="127.0.0.1", help="address to serve the page on (0.0.0.0 = all)")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    ap.add_argument("--pi-view-port", type=int, default=PI_VIEW_PORT,
                    help="port of mirage_vision's browser view on the Pi (its -p)")
    ap.add_argument("--allow-control", default="", metavar="IP[,IP...]",
                    help="other devices (with --host 0.0.0.0) that may use New reference frame and "
                         "Open valve; this computer always may")
    args = ap.parse_args()

    tel_sock, img_sock = bind_udp(TEL_PORT), bind_udp(IMG_PORT)
    console, sim, stop = Console(local_addresses()), Simulator(), threading.Event()
    for sock, size, handler in ((tel_sock, TEL_BUF, console.on_telemetry), (img_sock, IMG_BUF, console.on_image)):
        threading.Thread(target=receive, args=(sock, size, handler, stop), daemon=True).start()
    try:
        httpd = make_server(console, sim, args.host, args.port, stop, args.pi_view_port,
                            [ip.strip() for ip in args.allow_control.split(",") if ip.strip()])
    except OSError as e:
        raise SystemExit(f"Can't serve on {args.host}:{args.port} ({e.strerror}). Try --port {args.port + 1}.")

    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"Mirage console at {url}")
    print(f"listening on UDP {TEL_PORT} (telemetry) and {IMG_PORT} (thumbnails); "
          "dashboard.py can't run at the same time. Ctrl-C to quit.")
    if console.host_ips:
        print(f"this Mac: {console.host_ips[0]} (give it to ./supervisor and mirage_vision -i)")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sim.stop()
        httpd.server_close()
        tel_sock.close()
        img_sock.close()


if __name__ == "__main__":
    main()
