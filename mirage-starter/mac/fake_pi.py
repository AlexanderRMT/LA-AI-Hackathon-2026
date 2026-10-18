#!/usr/bin/env python3
"""fake_pi.py: pretends to be the Pi so the dashboard can be built and rehearsed alone.   owner: Gio

Sends the supervisor's exact telemetry JSON (UDP 5005, 10 Hz) and shift-map JPEG thumbnails
(UDP 5006) on a 30 s loop that walks through the chaos demo: boot closed, reset, a gas puff
trips the valve, reset while clear, vision stalls and the deadline trips it, reset, then a heat
source that must not trip it.

  python fake_pi.py                                   to this Mac (run dashboard.py alongside)
  python fake_pi.py --replay recordings/gas_*.npz     real shift maps from bos_proto.py clips
  python fake_pi.py --ip 172.20.10.3                  to another machine
"""
import argparse
import glob
import json
import socket
import time

import cv2
import numpy as np

DEADLINE_MS, TRIP_RUN, RESET_CLEAR, P_GAS_TRIP = 250, 3, 5, 0.60   # common/mirage_msgs.h
PHYSICS_GAS_PX = 0.30                                              # vision/mirage_hook.cpp fallback
CLEAR, GAS, HEAT, MOTION = range(4)
G = 64
yy, xx = np.mgrid[0:G, 0:G].astype(np.float32)


def blob_field(cx, cy, sx, sy, amp):
    """Shift field of a refracting blob: dots move along the gradient of its density."""
    g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
    gy, gx = np.gradient(g)
    s = amp / max(1e-6, float(np.hypot(gx, gy).max()))
    return np.dstack([gx * s, gy * s]).astype(np.float32)


def scripted(t, rng):
    """(cls, p_gas, field) at loop time t, or None while vision is stalled; plus a reset press."""
    noise = (0.03 * rng.standard_normal((G, G, 2))).astype(np.float32)
    reset = any(abs(t - r) < 0.01 for r in (2.0, 14.0, 22.0))
    if 18.0 <= t < 19.5:
        return None, reset                                       # vision stalls: no verdicts
    if 8.0 <= t < 10.5:                                          # lighter puff, gas sinks
        ph = t - 8.0
        f = blob_field(32 + 3 * np.sin(5 * t), 14 + 12 * ph, 5 + 3 * ph, 4 + 4 * ph, 0.7 * np.exp(-ph / 2))
        return (GAS, 0.93, f + noise), reset
    if 23.0 <= t < 29.0:                                         # soldering iron, heat rises
        f = sum(blob_field(32 + 4 * np.sin(9 * t + k), 60 - 50 * ((2 * t + k / 3) % 1), 4, 6,
                           0.35 * (1 - (2 * t + k / 3) % 1)) for k in range(3))
        return (HEAT, 0.06, f + noise), reset
    return (CLEAR, 0.02, noise), reset


def thumbnail(field, width=360, aspect=9 / 16):
    """Same picture as BosEngine::visual(): 1 px of shift = full brightness, inferno colors."""
    mag = np.hypot(field[..., 0], field[..., 1])
    m8 = np.clip(mag * 255.0, 0, 255).astype(np.uint8)
    img = cv2.resize(cv2.applyColorMap(m8, cv2.COLORMAP_INFERNO), (width, int(width * aspect)),
                     interpolation=cv2.INTER_CUBIC)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--replay", nargs="+", help="bos_proto.py .npz clips to play instead of the script")
    args = ap.parse_args()
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = np.random.default_rng(0)

    replay = []
    for pattern in args.replay or []:
        for path in sorted(glob.glob(pattern)):
            replay.extend(np.load(path)["fields"].astype(np.float32))
    if args.replay and not replay:
        raise SystemExit("no fields found in --replay files")

    S = dict(tripped=1, reason="boot", seq=0, cls=0, p_gas=0.0, peak=0.0, lat=0.0, lat_max=0.0,
             d2t=0.0, n=0, gas_run=0, clear_run=0, gas_start=None, last_verdict=None)
    start = time.monotonic()
    i = 0
    print(f"sending telemetry to {args.ip}:5005 and thumbnails to {args.ip}:5006 (Ctrl-C to stop)")
    while True:
        tick = start + i * 0.1
        time.sleep(max(0.0, tick - time.monotonic()))
        now = time.monotonic()
        if replay:
            f = replay[i % len(replay)]
            peak = float(np.hypot(f[..., 0], f[..., 1]).max())
            gas = peak > PHYSICS_GAS_PX
            verdict, reset = (GAS if gas else CLEAR, 0.9 if gas else 0.05, f), S["clear_run"] >= 30
        else:
            verdict, reset = scripted((i % 300) / 10.0, rng)       # 30 s loop, one step per verdict

        if verdict is not None:                                   # what the supervisor does per verdict
            cls, p_gas, field = verdict
            lat = 28.0 + 6.0 * rng.random()                       # frame timestamp to supervisor, ms
            frame_t = now - lat / 1000.0
            S.update(seq=S["seq"] + 1, cls=cls, p_gas=p_gas, lat=lat, lat_max=max(S["lat_max"], lat),
                     peak=float(np.hypot(field[..., 0], field[..., 1]).max()), n=S["n"] + 1, last_verdict=now)
            is_gas = cls == GAS or p_gas >= P_GAS_TRIP
            if is_gas:
                S["gas_start"] = S["gas_start"] or frame_t
                S["gas_run"] += 1
                S["clear_run"] = 0
            else:
                S["gas_run"], S["gas_start"] = 0, None
                S["clear_run"] += 1
            if not S["tripped"] and S["gas_run"] >= TRIP_RUN:
                S.update(tripped=1, reason="leak", d2t=(now - S["gas_start"]) * 1000.0)
            jpg = thumbnail(field)
            if jpg:
                tel.sendto(jpg, (args.ip, 5006))
        if reset and S["tripped"] and S["clear_run"] >= RESET_CLEAR:
            S.update(tripped=0, reason="none")
        since = (now - S["last_verdict"]) * 1000.0 if S["last_verdict"] else -1.0
        if not S["tripped"] and since > DEADLINE_MS:
            S.update(tripped=1, reason="deadline")

        msg = {"t_ms": int(now * 1000), "valve": "closed" if S["tripped"] else "open", "tripped": S["tripped"],
               "reason": S["reason"], "seq": S["seq"], "cls": S["cls"], "p_gas": round(S["p_gas"], 3),
               "peak_px": round(S["peak"], 3), "since_verdict_ms": round(since, 1), "lat_ms": round(S["lat"], 1),
               "lat_max_ms": round(S["lat_max"], 1), "detect_to_trip_ms": round(S["d2t"], 1), "verdicts": S["n"]}
        tel.sendto(json.dumps(msg, separators=(",", ":")).encode(), (args.ip, 5005))
        i += 1


if __name__ == "__main__":
    main()
