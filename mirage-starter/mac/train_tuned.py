#!/usr/bin/env python3
"""train_tuned.py: train.py plus a held-out check and the settings tuned on the Pi dumps.  owner: Gio

Same windows, preprocessing, model and TFLite export as train.py (imported from it, so the two
can't drift apart), and the same files for the Pi: export/model.tflite, labels.json,
model_info.json. train.py is unchanged and still works as before.

What it adds:
  --check      Scores every clip with a model trained without it, through the supervisor's own
               rule: a frame is gas when the top class is gas or p_gas >= 0.60, and 3 gas frames
               in a row close the valve. Quote these numbers. train.py validates on the last 20%
               of clips it also trained on, which flatters the model.
  presses.txt  A gas clip folder may list its press start times (seconds after the first field,
               one per line). Then a frame is gas only while a press is on and the shift shows;
               frames that show a shift between presses (lingering gas or noise) and frames with
               something solid in the box (over --solid-px: the lighter or a hand) are left out
               of training. Without the file, gas frames are labeled as in train.py.
  --zoom       Adds a synthetic table tap or camera drift to that share of the training windows:
               after median bump removal both leave a smooth zoom across the whole box, which
               the model must not call gas. The window keeps its label, since a tap doesn't
               change what is in the box. (Relabeling strong taps as motion was worse: the model
               then took motion to mean "zoom" and called people walking past gas.)
  --gas-bias   Added to the gas output's bias in the last layer; below 0 makes gas verdicts
               rarer. Same ops and file format, so the Pi loads it like any other model.

Defaults are what the held-out check picked on the 15 Pi clips of 2026-10-18 (Mac M1, about 8 min).

  python mac/train_tuned.py --data pi_dumps --check     held-out check, then train and export
  python mac/train_tuned.py --data pi_dumps             train on every clip and export
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                                    # noqa: E402

P_GAS_TRIP, TRIP_RUN = 0.60, 3                       # MIRAGE_P_GAS_TRIP, MIRAGE_TRIP_RUN in common/mirage_msgs.h
GAS_ON = (0.5, 3.5)                                  # gas is in the box 0.5 s to 3.5 s after a 2 s press starts
FOLDS = 5


def load(roots):
    """All windows in one array (prepared as train.py does), plus per-clip labels and timing."""
    clips = T.load_clips(roots)
    total = sum(max(0, len(c[1]) - (T.STACK - 1)) for c in clips)
    X = np.empty((total, T.GRID, T.GRID, 2 * T.STACK), np.float32)
    end, cid, meta, pos = np.empty(total, np.int64), np.empty(total, np.int64), [], 0
    for k, (label, fields, peak, src, hz, kind) in enumerate(clips):
        Xc, _ = T.windows(label, fields, peak, 0, len(fields))
        X[pos:pos + len(Xc)] = T.prep(Xc)
        end[pos:pos + len(Xc)] = np.arange(T.STACK - 1, T.STACK - 1 + len(Xc))
        cid[pos:pos + len(Xc)] = k
        pos += len(Xc)
        pf = os.path.join(src, "presses.txt")
        presses = np.loadtxt(pf, ndmin=1) if label == "gas" and os.path.isfile(pf) else None
        meta.append({"label": label, "peak": peak, "name": os.path.basename(src.rstrip("/")), "presses": presses})
    return X, end, cid, meta, clips


def frame_labels(m, active, solid_px=float("inf")):
    """(label, weight) for every frame of one clip; weight 0 leaves the frame out of training."""
    label, peak = m["label"], m["peak"]
    y = np.full(len(peak), T.LABELS.index(label), np.int64)
    w = np.ones(len(peak), np.float32)
    if label not in active:
        return y, w
    strong = peak >= active[label]
    if m["presses"] is None:
        y[~strong] = 0                               # train.py's rule
        return y, w
    t = np.arange(len(peak)) / T.PROC_HZ
    on = np.zeros(len(peak), bool)
    for p in m["presses"]:
        on |= (t >= p + GAS_ON[0]) & (t < p + GAS_ON[1])
    y[~(on & strong)] = 0
    w[~on & strong] = 0.0
    w[peak > solid_px] = 0.0
    return y, w


def all_labels(end, cid, meta, active, solid_px=float("inf")):
    y, w = np.empty(len(end), np.int64), np.empty(len(end), np.float32)
    for k, m in enumerate(meta):
        yk, wk = frame_labels(m, active, solid_px)
        sel = cid == k
        y[sel], w[sel] = yk[end[sel]], wk[end[sel]]
    return y, w


def zoom_nuisance(rng, n, lo_px, hi_px):
    """Whole-box zoom fields like a table tap (a jolt that rings down) or a drifting camera leaves
    after median bump removal. Prepared units, plus each field's size at the box edge in px."""
    g = (np.arange(T.GRID) - (T.GRID - 1) / 2) / ((T.GRID - 1) / 2)     # -1..1 across the box
    xx, yy = np.meshgrid(g, g)                                          # xx along columns (x), yy along rows (y)
    m = np.exp(rng.uniform(np.log(lo_px), np.log(hi_px), n)) * rng.choice([-1.0, 1.0], n)
    zy = rng.uniform(0.6, 1.2, n)                                       # taps are mostly zoom, a bit uneven
    sxy, syx = rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n)     # and a little tilt or shear
    prof = np.zeros((n, T.STACK))
    for i in range(n):
        if rng.random() < 0.5:                                          # jolt at field k, then ringing
            k = rng.integers(0, T.STACK)
            prof[i, k] = 1.0
            for j in range(k + 1, T.STACK):
                prof[i, j] = prof[i, j - 1] * rng.uniform(0.1, 0.6) * rng.choice([-1.0, 1.0])
        else:                                                           # slow drift
            prof[i] = 1.0 + rng.uniform(-0.3, 0.3) * np.arange(T.STACK) / (T.STACK - 1)
            prof[i] /= np.abs(prof[i]).max()
    a = (m[:, None] * prof)[:, None, None, :]
    out = np.empty((n, T.GRID, T.GRID, 2 * T.STACK), np.float32)
    out[..., 0::2] = a * (xx[None, ..., None] + sxy[:, None, None, None] * yy[None, ..., None])
    out[..., 1::2] = a * (syx[:, None, None, None] * xx[None, ..., None] + zy[:, None, None, None] * yy[None, ..., None])
    return out * T.SCALE, np.abs(m)


def batches(X, idx, y, w, cw, args, seed, batch=64):
    """train.py's augmentation (mirror with dx negated, small noise) plus the synthetic taps."""
    import keras
    lim = T.CLIP_PX * T.SCALE

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(idx))

        def __len__(self):
            return int(np.ceil(len(idx) / batch))

        def __getitem__(self, i):
            sel = np.sort(idx[self.order[i * batch:(i + 1) * batch]])
            xb, yb, wb = X[sel], y[sel], w[sel]      # fancy indexing copies, X stays untouched
            flip = self.rng.random(len(sel)) < 0.5   # never up-down: sinking gas would look like rising heat
            if flip.any():
                xf = xb[flip][:, :, ::-1, :].copy()
                xf[..., 0::2] *= -1.0
                xb[flip] = xf
            if args.zoom > 0:
                z = np.flatnonzero(self.rng.random(len(sel)) < args.zoom)
                if len(z):
                    xb[z] += zoom_nuisance(self.rng, len(z), *args.zoom_px)[0]
                    np.clip(xb, -lim, lim, out=xb)
            xb += 0.02 * self.rng.standard_normal(xb.shape, dtype=np.float32)
            return xb, yb, cw[yb] * wb

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(idx))

    return Batches()


def fit(X, idx, y, w, args, verbose=0):
    """train.py's model and class balancing, trained on the windows in idx."""
    import keras
    keras.utils.set_random_seed(args.seed)
    idx = idx[::args.stride]
    idx = idx[w[idx] > 0]
    counts = np.bincount(y[idx], minlength=len(T.LABELS)).astype(np.float64)
    cw = np.array([counts.sum() / (len(T.LABELS) * c) if c > 0 else 0.0 for c in counts], np.float32)
    model = T.build_model()
    model.fit(batches(X, idx, y, w, cw, args, args.seed), epochs=args.epochs, verbose=verbose)
    add_gas_bias(model, args.gas_bias)
    return model, {T.LABELS[i]: int(c) for i, c in enumerate(counts)}


def add_gas_bias(model, delta):
    if delta:
        W, b = model.layers[-1].get_weights()
        b = b.copy()
        b[T.LABELS.index("gas")] += delta
        model.layers[-1].set_weights([W, b])


def run_lengths(gas):
    out, run = np.zeros(len(gas), np.int64), 0
    for i, g in enumerate(gas):
        run = run + 1 if g else 0
        out[i] = run
    return out


def score_clip(m, t, probs):
    """What the supervisor would have done with this clip's verdicts."""
    gas = (probs.argmax(1) == T.LABELS.index("gas")) | (probs[:, T.LABELS.index("gas")] >= P_GAS_TRIP)
    run = run_lengths(gas)
    r = {"clip": m["name"], "label": m["label"], "trips": int(np.sum(run == TRIP_RUN)), "gas_frames": float(gas.mean())}
    if m["label"] == "gas" and m["presses"] is not None:
        lat = []
        for p in m["presses"]:
            closed = t[(run >= TRIP_RUN) & (t >= p) & (t < p + 5.0)]
            if len(closed):
                lat.append(float(closed[0] - p))
        r.update({"presses": len(m["presses"]), "presses_hit": len(lat), "latency_s": float(np.median(lat)) if lat else None,
                  "before_first_press": int(np.sum((run == TRIP_RUN) & (t < m["presses"][0])))})
    return r


def check(X, end, cid, meta, y, w, args):
    """Every clip scored by a model trained without it. Folds take one clip of each label at a time."""
    fold = np.zeros(len(meta), np.int64)
    for label in T.LABELS:
        ks = sorted((k for k, m in enumerate(meta) if m["label"] == label), key=lambda k: meta[k]["name"])
        for i, k in enumerate(ks):
            fold[k] = i % FOLDS
    rows, cm, t0 = [], np.zeros((len(T.LABELS), len(T.LABELS)), np.int64), time.time()
    for f in range(FOLDS):
        held = [k for k in range(len(meta)) if fold[k] == f]
        if not held:
            continue
        model, _ = fit(X, np.flatnonzero(~np.isin(cid, held)), y, w, args)
        for k in held:
            te = np.flatnonzero(cid == k)
            probs = model.predict(X[te], batch_size=256, verbose=0)
            rows.append(score_clip(meta[k], end[te] / T.PROC_HZ, probs))
            keep = w[te] > 0
            np.add.at(cm, (y[te][keep], probs.argmax(1)[keep]), 1)
        print(f"  fold {f + 1}/{FOLDS} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    gas = [r for r in rows if r["label"] == "gas"]
    timed = [r for r in gas if "presses" in r]
    other = [r for r in rows if r["label"] != "gas"]
    lat = [r["latency_s"] for r in timed if r["latency_s"] is not None]
    s = {"gas_clips": len(gas), "gas_clips_tripped": sum(r["trips"] > 0 for r in gas),
         "presses": sum(r["presses"] for r in timed), "presses_hit": sum(r["presses_hit"] for r in timed),
         "median_latency_s": float(np.median(lat)) if lat else None,
         "trips_before_first_press": sum(r["before_first_press"] for r in timed),
         "other_clips": len(other), "false_trips": sum(r["trips"] for r in other),
         "false_gas_frames": float(np.mean([r["gas_frames"] for r in other])) if other else None,
         "clips": rows, "confusion": cm.tolist()}
    print("\nheld-out check (each clip scored by a model trained without it, through the supervisor's rule)")
    for r in rows:
        extra = (f"  presses {r['presses_hit']}/{r['presses']}, median {r['latency_s']} s" if "presses" in r else "")
        print(f"  {r['clip']:22s} {r['label']:6s} valve trips {r['trips']:3d}  gas frames {100 * r['gas_frames']:5.1f}%{extra}")
    print("\nconfusion matrix over held-out frames (rows = truth, columns = predicted):")
    print("        " + " ".join(f"{l:>7s}" for l in T.LABELS))
    for i, l in enumerate(T.LABELS):
        print(f"{l:>7s} " + " ".join(f"{v:7d}" for v in cm[i]))
    if timed:
        print(f"\ngas: {s['presses_hit']}/{s['presses']} presses closed the valve, median {s['median_latency_s']} s "
              f"after the press started; {s['gas_clips_tripped']}/{s['gas_clips']} gas clips closed it at least once")
    print(f"false trips: {s['false_trips']} on {s['other_clips']} clear, heat and motion clips "
          f"({100 * (s['false_gas_frames'] or 0):.2f}% of their frames read as gas)")
    return s


def export(model, args, active, counts, held_out, sample):
    os.makedirs(args.out, exist_ok=True)
    tfl = T.to_tflite(model, args.out)
    with open(os.path.join(args.out, "model.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(T.LABELS, f)
    info = {"input": [1, T.GRID, T.GRID, 2 * T.STACK], "dtype": "float32",
            "channels": "dx, dy for each field, oldest first",
            "clip_px": T.CLIP_PX, "scale": T.SCALE, "proc_hz": T.PROC_HZ, "labels": T.LABELS,
            "active_px": active, "ops": T.op_versions(tfl),
            "training": {"script": "mac/train_tuned.py", "data": args.data, "epochs": args.epochs, "seed": args.seed,
                         "stride": args.stride, "solid_px": args.solid_px, "zoom": args.zoom, "zoom_px": args.zoom_px,
                         "gas_bias": args.gas_bias, "gas_on_s": GAS_ON,
                         "train_counts": counts,
                         "held_out_check": {k: v for k, v in held_out.items() if k != "clips"} if held_out else None}}
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # The Pi runs exactly this file: check it with the TFLite interpreter against the Keras model.
    interp = T.interpreter(tfl)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    want = model.predict(sample, verbose=0)
    got = []
    for x in sample:
        interp.set_tensor(inp["index"], x[None])
        interp.invoke()
        got.append(interp.get_tensor(out["index"])[0].copy())
    diff = float(np.abs(np.array(got) - want).max())
    print(f"\nexported {args.out}/model.tflite ({len(tfl) / 1024:.0f} KB), input {inp['shape'].tolist()}, "
          f"output {out['shape'].tolist()}, TFLite vs Keras max difference {diff:.2e}")
    print(f"ops the Pi needs: {T.op_versions(tfl)}")
    if diff > 1e-3:
        raise SystemExit("TFLite output differs from the trained model; not safe to ship")


def pair(s):
    return tuple(float(v) for v in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["recordings"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth training window (2 halves memory)")
    ap.add_argument("--out", default="export")
    ap.add_argument("--check", action="store_true", help="held-out check first (about 6 minutes)")
    ap.add_argument("--solid-px", type=float, default=3.0)
    ap.add_argument("--zoom", type=float, default=0.3, help="share of training windows given a synthetic tap (0 = off)")
    ap.add_argument("--zoom-px", type=pair, default=(0.03, 0.6), help="smallest,largest tap size at the box edge")
    ap.add_argument("--gas-bias", type=float, default=-1.5, help="0 = the model as trained")
    ap.add_argument("--active", type=lambda s: {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))},
                    default=None, help="e.g. gas=0.12,heat=0.04,motion=0.10 (default: train.py's ACTIVE_PX)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    active = {**T.ACTIVE_PX, **(args.active or {})}
    T.ACTIVE_PX = active                             # so the data report shows the thresholds in use
    X, end, cid, meta, clips = load(args.data)
    if not meta:
        raise SystemExit("No data found. Record with bos_proto.py or copy Pi dumps here.")
    T.data_report(clips)
    del clips
    timed = [m["name"] for m in meta if m["presses"] is not None]
    print(f"{len(X)} windows from {len(meta)} clips ({X.nbytes / 1e9:.2f} GB); press timing for {len(timed)} gas clips")
    y, w = all_labels(end, cid, meta, active, args.solid_px)

    held_out = check(X, end, cid, meta, y, w, args) if args.check else None
    print("\ntraining on every clip", flush=True)
    model, counts = fit(X, np.arange(len(X)), y, w, args, verbose=2)
    print("train counts:", counts)
    pick = np.random.default_rng(0).choice(len(X), size=min(32, len(X)), replace=False)
    export(model, args, active, counts, held_out, X[np.sort(pick)])


if __name__ == "__main__":
    main()
