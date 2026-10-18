#!/usr/bin/env python3
"""test_web_dashboard.py: web_dashboard.py must read packets exactly like dashboard.py.   owner: Gio

Replays the same packets, frame by frame, through dashboard.py's real main loop (window,
clock and sockets stubbed out) and through web_dashboard.Console, then compares every line
of text and every history bar dashboard.py would have drawn. Also checks the page server's
simulator guard. No network, no windows.

  python mac/test_web_dashboard.py        (or: pytest mac/test_web_dashboard.py)
"""
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard  # noqa: E402
import web_dashboard as web  # noqa: E402

FRAME_S = 0.033                                  # dashboard.py redraws about every 30 ms
T0 = 1_760_000_000.0                             # a real time.time(): dashboard.py starts with last_rx = 0.0
ADDR_TEL, ADDR_IMG = ("192.168.1.87", 40000), ("192.168.1.87", 40001)


def packet(**fields):
    return json.dumps(fields).encode()


def supervisor_packet(t_ms, **over):
    msg = {"t_ms": t_ms, "valve": "open", "tripped": 0, "reason": "none", "seq": t_ms // 100, "cls": 0,
           "p_gas": 0.02, "peak_px": 0.031, "since_verdict_ms": 40.0, "lat_ms": 30.0, "lat_max_ms": 41.5,
           "detect_to_trip_ms": 0.0, "verdicts": t_ms // 100}
    msg.update(over)
    return packet(**msg)


def scenario():
    """[(t, telemetry datagrams, image datagrams)] per dashboard frame, covering every display path."""
    rng = np.random.default_rng(7)
    ok, jpg = cv2.imencode(".jpg", rng.integers(0, 255, (90, 160, 3), dtype=np.uint8))
    assert ok
    frames, t, t_ms = [], T0, 1000

    def add(n_frames, make=None, every=3, img_every=0):
        nonlocal t, t_ms
        for i in range(n_frames):
            tel = []
            if make and i % every == 0:
                tel.append(make(t_ms))
                t_ms += 100
            img = [jpg.tobytes()] if img_every and i % img_every == 0 else []
            frames.append((t, tel, img))
            t += FRAME_S

    add(6)                                                            # nothing yet
    add(30, lambda m: supervisor_packet(m, valve="closed", tripped=1, reason="boot", seq=0, cls=0, p_gas=0.0,
                                        peak_px=0.0, since_verdict_ms=-1.0, lat_ms=0.0, lat_max_ms=0.0,
                                        verdicts=0))
    add(60, lambda m: supervisor_packet(m, lat_ms=float(rng.uniform(20, 40))), img_every=3)
    add(12, lambda m: supervisor_packet(m, cls=1, p_gas=0.93, peak_px=0.61), img_every=3)
    add(60, lambda m: supervisor_packet(m, valve="closed", tripped=1, reason="leak", cls=1, p_gas=0.95,
                                        detect_to_trip_ms=231.4, lat_ms=float(rng.uniform(20, 40))))
    add(30, lambda m: supervisor_packet(m, valve="closed", tripped=1, reason="leak", detect_to_trip_ms=231.4))
    add(30, lambda m: supervisor_packet(m, cls=2, p_gas=0.06, detect_to_trip_ms=231.4))     # heat
    add(30, lambda m: supervisor_packet(m, cls=3, p_gas=0.11))                              # motion
    add(9, lambda m: supervisor_packet(m, cls=7))                                           # unknown class
    add(9, lambda m: packet(valve="open", tripped=0))                                       # most fields missing
    add(9, lambda m: supervisor_packet(m, lat_ms=0.0))                                      # no latency measured
    add(30, lambda m: supervisor_packet(m, valve="closed", tripped=1, reason="deadline",
                                        since_verdict_ms=260.0 + (m % 1000)))
    frames.append((t, [b"{not json", b"\xff\xfe", b"", supervisor_packet(t_ms)], [b"not a jpeg"]))
    last_rx = t
    t += FRAME_S
    for dt in (0.5, 0.99, 1.0, 1.001, 1.5, 4.0):                     # the stale boundary at 1.0 s
        frames.append((last_rx + dt, [], []))
    t = last_rx + 4.0 + FRAME_S
    add(700, lambda m: supervisor_packet(m, lat_ms=float(rng.uniform(10, 90)),
                                         p_gas=float(rng.uniform(0, 1))), every=1)          # fills both deques
    return frames


class FakeSock:
    def __init__(self, addr):
        self.queue, self.addr = [], addr

    def recvfrom(self, n):
        if not self.queue:
            raise BlockingIOError
        return self.queue.pop(0)[:n], self.addr


def run_dashboard(frames):
    """Drives dashboard.main() frame by frame; returns what it drew in each frame."""
    socks = {5005: FakeSock(ADDR_TEL), 5006: FakeSock(ADDR_IMG)}
    clock, idx, drawn = [0.0], [0], []
    current = {"text": [], "bars": []}

    def load(i):
        clock[0] = frames[i][0]
        socks[5005].queue.extend(frames[i][1])
        socks[5006].queue.extend(frames[i][2])

    def put_text(_img, text, *a, **k):
        current["text"].append(text)

    def rectangle(_img, p0, p1, color, *a, **k):
        current["bars"].append((p1[1] - p0[1], tuple(color) == (60, 70, 230)))

    def wait_key(_ms):
        drawn.append(dict(current))
        current["text"], current["bars"] = [], []
        idx[0] += 1
        if idx[0] == len(frames):
            return ord("q")
        load(idx[0])
        return -1

    load(0)
    with mock.patch.object(dashboard, "listener", lambda port: socks[port]), \
            mock.patch.object(dashboard, "time", types.SimpleNamespace(time=lambda: clock[0])), \
            mock.patch.object(dashboard.cv2, "imshow", lambda *a: None), \
            mock.patch.object(dashboard.cv2, "putText", put_text), \
            mock.patch.object(dashboard.cv2, "rectangle", rectangle), \
            mock.patch.object(dashboard.cv2, "waitKey", wait_key), \
            mock.patch.object(dashboard.cv2, "destroyAllWindows", lambda: None), \
            mock.patch("builtins.print", lambda *a, **k: None):
        dashboard.main()
    return drawn


def legacy_view(snap):
    """dashboard.py's text and bars, rebuilt from web_dashboard's snapshot with the same formats."""
    tel, state = snap["telemetry"], snap["telemetry"]["state"]
    text = ["VALVE OPEN" if tel["valve_open"] else "VALVE CLOSED",
            "no telemetry" if tel["stale"] else f"reason: {state.get('reason', '?')}",
            f"class: {tel['class_label']}   p(gas) {state.get('p_gas', 0):.2f}",
            f"peak shift: {state.get('peak_px', 0):.2f} px",
            f"since last verdict: {state.get('since_verdict_ms', 0):.0f} ms (limit 250)",
            f"frame to supervisor: {state.get('lat_ms', 0):.1f} ms (max {state.get('lat_max_ms', 0):.1f})",
            f"first gas frame to valve: {state.get('detect_to_trip_ms', 0):.0f} ms"]
    if tel["lat_count"]:
        text.append(f"latency p99 over last {tel['lat_count']}: {tel['lat_p99']:.1f} ms")
    bars = [(int(p * 120), bool(tripped)) for p, tripped, _ in tel["history"]]
    return text, bars


def test_same_display_as_dashboard():
    frames = scenario()
    drawn = run_dashboard(frames)
    assert len(drawn) == len(frames)
    console = web.Console()
    for i, (t, tel, img) in enumerate(frames):
        for d in tel:
            console.on_telemetry(d[:web.TEL_BUF], ADDR_TEL, t)
        for d in img:
            console.on_image(d[:web.IMG_BUF], ADDR_IMG, t)
        text, bars = legacy_view(console.snapshot(t))
        assert drawn[i]["text"] == text, f"frame {i} at t={t:.3f}:\n  dashboard {drawn[i]['text']}\n  web       {text}"
        assert drawn[i]["bars"] == bars, f"frame {i}: history differs"
    snap = console.snapshot(frames[-1][0])["telemetry"]
    assert len(snap["history"]) == web.HISTORY_LEN and snap["lat_count"] == web.LAT_LEN
    assert snap["malformed"] == 3


def test_stale_boundary():
    console = web.Console()
    console.on_telemetry(supervisor_packet(1000), ADDR_TEL, 10.0)
    assert console.snapshot(10.0)["telemetry"]["valve_open"]
    assert not console.snapshot(11.0)["telemetry"]["stale"]                # exactly 1.0 s: still fresh
    snap = console.snapshot(11.001)["telemetry"]
    assert snap["stale"] and not snap["valve_open"] and snap["state"]["valve"] == "open"


def test_same_ports_and_buffers_as_dashboard():
    src = Path(dashboard.__file__).read_text()
    assert (web.TEL_PORT, web.IMG_PORT) == (5005, 5006) and "listener(5005), listener(5006)" in src
    assert f"recvfrom({web.TEL_BUF})" in src and f"recvfrom({web.IMG_BUF})" in src
    assert web.LABELS == dashboard.LABELS
    assert "maxlen=120" in src and "maxlen=600" in src and "> 1.0" in src


def test_images_kept_only_when_they_decode():
    console = web.Console()
    ok, jpg = cv2.imencode(".jpg", np.zeros((202, 360, 3), np.uint8))
    assert console.on_image(jpg.tobytes(), ADDR_IMG, 1.0)
    assert not console.on_image(b"\xff\xd8 broken", ADDR_IMG, 1.1)
    data, seq = console.thumbnail()
    snap = console.snapshot(1.2)["image"]
    assert data == jpg.tobytes() and seq == 1 and snap["rejected"] == 1
    assert (snap["width"], snap["height"]) == (360, 202)


def test_odd_packets_never_crash_or_leak_nan():
    console = web.Console()
    for d in (b"[1, 2]", b'"text"', b"42", b"null"):
        assert not console.on_telemetry(d, ADDR_TEL, 1.0)
    assert console.on_telemetry(b'{"p_gas": NaN, "cls": "1", "lat_ms": "x"}', ADDR_TEL, 1.0)
    snap = console.snapshot(1.0)
    json.dumps(snap, allow_nan=False)
    assert snap["telemetry"]["class_label"] == "?" and snap["telemetry"]["malformed"] == 4


def test_simulator_detection_and_guard():
    console = web.Console(["192.168.1.86"])
    console.on_telemetry(supervisor_packet(1000), ("127.0.0.1", 50000), 1.0)
    snap = console.snapshot(1.0)["telemetry"]
    assert snap["simulated"] and not snap["multi_source"]
    assert "already sending" in console.sim_block_reason(1.0, False)
    console.on_telemetry(supervisor_packet(1100), ("192.168.1.87", 50001), 1.1)
    snap = console.snapshot(1.1)["telemetry"]
    assert snap["multi_source"] and not snap["simulated"]
    assert "192.168.1.87" in console.sim_block_reason(1.1, False)
    assert console.sim_block_reason(7.0, False) is None                    # quiet for 5 s: allowed
    assert web.contract_constants()["source"] == "common/mirage_msgs.h"
    assert {k: web.contract_constants()[k] for k in ("deadline_ms", "trip_run", "reset_clear", "p_gas_trip")} == \
        {"deadline_ms": 250, "trip_run": 3, "reset_clear": 5, "p_gas_trip": 0.60}


def test_page_server_and_control_guard():
    console, stop = web.Console(), threading.Event()

    class NoSim:
        started = False

        def running(self):
            return False

        def pid(self):
            return None

        def start(self):
            NoSim.started = True

        def stop(self):
            pass

    httpd = web.make_server(console, NoSim(), "127.0.0.1", 0, stop)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        snap = json.load(urllib.request.urlopen(base + "/api/state", timeout=3))
        assert snap["telemetry"]["received"] is False and snap["sim"]["running"] is False
        assert urllib.request.urlopen(base + "/", timeout=3).status == 200

        def post(path, headers):
            req = urllib.request.Request(base + path, method="POST", headers=headers)
            try:
                return urllib.request.urlopen(req, timeout=3).status
            except urllib.error.HTTPError as e:
                return e.code
        assert post("/api/sim/start", {}) == 403                                    # no control header
        assert post("/api/sim/start", {"X-Mirage-Control": "1", "Origin": "http://evil.example"}) == 403
        assert not NoSim.started
        assert post("/api/sim/start", {"X-Mirage-Control": "1"}) == 200 and NoSim.started
        assert post("/api/nope", {"X-Mirage-Control": "1"}) == 404
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:                   # noqa: BLE001 - report every failure, then exit 1
            failed += 1
            print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
