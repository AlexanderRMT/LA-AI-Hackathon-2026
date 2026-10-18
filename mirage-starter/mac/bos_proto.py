#!/usr/bin/env python3
"""bos_proto.py: background-oriented schlieren on a Mac.                 owner: Gio

Proves the physics before the Pi pipeline is ready and records early training data.
It copies the processing steps in vision/bos_engine.cpp, so keep the constants in sync.

  pip install opencv-python numpy
  python bos_proto.py                       Mac camera
  python bos_proto.py --cam 1               another camera (iPhone Continuity Camera, USB)
  python bos_proto.py --video lighter.mov   a phone video
  python bos_proto.py --video lighter.mov --headless --label gas    batch-convert to training data

Keys: r new reference   0 clear  1 gas  2 heat  3 motion (sets the label)
      space start/stop recording   q quit
Setup: point the camera at the dot background (Mirage page, "Use this screen as the dot
background"), hold everything still, press r, then puff an unlit lighter about halfway
between the camera and the screen.
"""
import argparse
import os
import time

import cv2
import numpy as np

GRID, EMA, PROC_HZ = 64, 0.5, 10.0          # must match BosParams in vision/bos_engine.hpp
LABELS = ["clear", "gas", "heat", "motion"]
FULL_SCALE_PX = 1.0                         # 1 px of shift = brightest color on screen


def save_clip(out_dir, label, fields, times, peaks):
    if len(fields) < 8:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_{time.strftime('%H%M%S')}_{time.time_ns() % 1000000:06d}.npz")
    np.savez_compressed(path, fields=np.asarray(fields, np.float32), t=np.asarray(times),
                        peak=np.asarray(peaks, np.float32), label=label)
    print(f"saved {path} ({len(fields)} fields)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--label", default="clear", choices=LABELS)
    ap.add_argument("--headless", action="store_true", help="no windows; record the whole video with --label")
    ap.add_argument("--roi", help="x,y,w,h of the dot screen; skips the selection window")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.width * 9 // 16)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No frames. On macOS, give your terminal app camera access in System Settings.")

    h_full, w_full = frame.shape[:2]
    roi = (0, 0, w_full, h_full)
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
    elif not args.headless:
        title = "Drag a box around the dot screen, then press Enter"
        r = cv2.selectROI(title, frame, showCrosshair=False)
        cv2.destroyWindow(title)
        if r[2] > 16 and r[3] > 16:
            roi = tuple(int(v) for v in r)
    x, y, w, h = roi
    print(f"frame {w_full}x{h_full}, dot screen ROI {roi}")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0.0
    ref, smooth = None, None
    label, recording = args.label, args.headless
    fields, times, peaks = [], [], []
    last_proc, frame_idx, t0 = -1e9, 0, time.time()
    fps_count, fps_t, fps = 0, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / video_fps if (args.video and video_fps > 0) else time.time() - t0
        if ref is not None and t - last_proc < 1.0 / PROC_HZ - 1e-6:
            continue                                   # fixed processing rate, same as the Pi
        last_proc = t
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        if ref is None:
            ref, smooth = gray.copy(), None
            continue

        flow = dis.calc(ref, gray, None)               # (h, w, 2) float32, pixels
        bump = np.median(flow[::4, ::4].reshape(-1, 2), axis=0)
        flow -= bump                                   # remove table bumps
        small = cv2.resize(flow, (GRID, GRID), interpolation=cv2.INTER_AREA)
        smooth = small if smooth is None else (EMA * small + (1.0 - EMA) * smooth).astype(np.float32)
        peak = float(np.linalg.norm(smooth, axis=2).max())

        if recording:
            fields.append(smooth.copy())
            times.append(t)
            peaks.append(peak)

        fps_count += 1
        if time.time() - fps_t >= 1.0:
            fps, fps_count, fps_t = fps_count / (time.time() - fps_t), 0, time.time()

        if args.headless:
            continue
        mag = cv2.GaussianBlur(np.linalg.norm(flow, axis=2), (0, 0), 2.0)
        vis = cv2.applyColorMap(np.clip(mag / FULL_SCALE_PX * 255.0, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_INFERNO)
        scale = 720.0 / max(w, h)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cam = cv2.resize(gray, (int(w * scale), int(h * scale)))
        cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2BGR)
        status = f"peak {peak:.2f} px  bump {np.hypot(*bump):.2f} px  {fps:.0f} Hz  label {label}"
        status += "  REC" if recording else ""
        cv2.putText(vis, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Mirage prototype: camera | shift map", np.hstack([cam, vis]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ref, smooth = gray.copy(), None
            print("new reference")
        if key in (ord("0"), ord("1"), ord("2"), ord("3")):
            if recording:
                save_clip(args.out, label, fields, times, peaks)
                fields, times, peaks = [], [], []
            label = LABELS[key - ord("0")]
            print("label:", label)
        if key == ord(" "):
            if recording:
                save_clip(args.out, label, fields, times, peaks)
                fields, times, peaks = [], [], []
            recording = not recording
            print("recording" if recording else "stopped")

    if recording:
        save_clip(args.out, label, fields, times, peaks)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
