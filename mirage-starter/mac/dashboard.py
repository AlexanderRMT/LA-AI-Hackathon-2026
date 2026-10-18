#!/usr/bin/env python3
"""dashboard.py: live view of what the Pi is doing.                         owner: Gio

Listens for the supervisor's JSON telemetry (UDP 5005) and the vision process's JPEG
shift-map thumbnails (UDP 5006). Start the supervisor as `./supervisor <this Mac's IP>` and
pass the same IP to mirage_init() on the Pi.

  pip install opencv-python numpy
  python dashboard.py
  python fake_pi.py           in another terminal: fake telemetry, no Pi needed
"""
import collections
import json
import socket
import time

import cv2
import numpy as np

W, H = 960, 540
LABELS = ["clear", "gas", "heat", "motion"]


def listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.setblocking(False)
    return s


def main():
    tel_sock, img_sock = listener(5005), listener(5006)
    state, thumb, last_rx = {}, None, 0.0
    history = collections.deque(maxlen=120)           # 12 s of verdicts at 10 Hz
    lat = collections.deque(maxlen=600)
    print("listening on UDP 5005 (telemetry) and 5006 (thumbnails). Press q to quit.")
    while True:
        for _ in range(50):
            try:
                data, _ = tel_sock.recvfrom(2048)
            except BlockingIOError:
                break
            try:
                state = json.loads(data.decode())
            except ValueError:
                continue                                # a stray or truncated packet
            last_rx = time.time()
            history.append((state.get("p_gas", 0.0), state.get("tripped", 1)))
            if state.get("lat_ms", 0) > 0:
                lat.append(state["lat_ms"])
        for _ in range(10):
            try:
                data, _ = img_sock.recvfrom(65535)
            except BlockingIOError:
                break
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                thumb = img

        canvas = np.full((H, W, 3), (26, 20, 14), np.uint8)
        if thumb is not None:
            th = cv2.resize(thumb, (560, int(560 * thumb.shape[0] / thumb.shape[1])))
            canvas[20:20 + th.shape[0], 20:20 + th.shape[1]] = th[: H - 20, :]
        stale = time.time() - last_rx > 1.0
        valve_open = state.get("valve") == "open" and not stale
        color = (90, 200, 120) if valve_open else (60, 70, 230)
        x0 = 600
        cv2.putText(canvas, "VALVE OPEN" if valve_open else "VALVE CLOSED", (x0, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        lines = [
            "no telemetry" if stale else f"reason: {state.get('reason', '?')}",
            f"class: {LABELS[state['cls']] if 'cls' in state and state['cls'] < 4 else '?'}"
            f"   p(gas) {state.get('p_gas', 0):.2f}",
            f"peak shift: {state.get('peak_px', 0):.2f} px",
            f"since last verdict: {state.get('since_verdict_ms', 0):.0f} ms (limit 250)",
            f"frame to supervisor: {state.get('lat_ms', 0):.1f} ms (max {state.get('lat_max_ms', 0):.1f})",
            f"first gas frame to valve: {state.get('detect_to_trip_ms', 0):.0f} ms",
        ]
        if lat:
            p99 = float(np.percentile(np.asarray(lat), 99))
            lines.append(f"latency p99 over last {len(lat)}: {p99:.1f} ms")
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (x0, 120 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 232, 238), 1)
        # p(gas) strip: one bar per verdict, red while tripped
        y0, bw = H - 30, max(1, (W - 40) // max(1, history.maxlen))
        cv2.line(canvas, (20, y0), (W - 20, y0), (90, 90, 90), 1)
        for i, (p, tripped) in enumerate(history):
            h = int(p * 120)
            cv2.rectangle(canvas, (20 + i * bw, y0 - h), (20 + i * bw + bw - 1, y0),
                          (60, 70, 230) if tripped else (90, 200, 120), -1)
        cv2.imshow("Mirage dashboard", canvas)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
