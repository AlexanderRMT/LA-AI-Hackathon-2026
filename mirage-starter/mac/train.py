#!/usr/bin/env python3
"""train.py: trains the Mirage classifier and exports model.tflite.        owner: Gio

Data it reads (any mix, searched recursively):
  *.npz                 from bos_proto.py (fields, peak, label)
  <label>_<anything>/   Pi dumps from BosEngine::startDump: fields.f32 + meta.csv,
                        where the folder name starts with clear, gas, heat or motion
Writes export/model.tflite, export/labels.json, export/model_info.json.

  pip install tensorflow numpy
  python train.py --data recordings pi_dumps
  python train.py --data pi_dumps --stride 2        half the windows if memory is tight

Before training it prints a data report: processing rate per clip (should be 10 Hz), peak
shift, how many quiet frames were relabeled clear, and a warning when an ACTIVE_PX
threshold sits below the noise floor measured on the clear clips.
"""
import argparse
import glob
import json
import os

import numpy as np

GRID, STACK, CLIP_PX, SCALE, PROC_HZ = 64, 4, 4.0, 0.5, 10.0   # must match BosParams
LABELS = ["clear", "gas", "heat", "motion"]                    # must match enum mirage_class
# Frames inside an active clip that barely move are really "clear" (between lighter puffs, say).
# Check these against the noise floor in the data report; Pi and Mac cameras differ.
ACTIVE_PX = {"gas": 0.12, "heat": 0.04, "motion": 0.10}


def load_clips(roots):
    """Returns (label, fields, peak, src, hz, kind) per clip; kind is 'mac' or 'pi'."""
    clips = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
            d = np.load(f, allow_pickle=False)
            t = d["t"].astype(np.float64) if "t" in d.files else np.zeros(0)
            hz = 1.0 / float(np.median(np.diff(t))) if t.size > 2 and np.median(np.diff(t)) > 0 else float("nan")
            clips.append((str(d["label"]), d["fields"].astype(np.float32), d["peak"].astype(np.float32), f, hz, "mac"))
        for meta in sorted(glob.glob(os.path.join(root, "**", "meta.csv"), recursive=True)):
            folder = os.path.dirname(meta)
            name = os.path.basename(folder).lower()
            label = next((l for l in LABELS if name.startswith(l)), None)
            if label is None:
                print(f"skipping {folder}: its name must start with one of {LABELS}")
                continue
            raw = np.fromfile(os.path.join(folder, "fields.f32"), dtype=np.float32)
            n = raw.size // (GRID * GRID * 2)
            fields = raw[: n * GRID * GRID * 2].reshape(n, GRID, GRID, 2)
            m = np.loadtxt(meta, delimiter=",", skiprows=1, ndmin=2)
            peak = m[:n, 2].astype(np.float32) if m.size else np.linalg.norm(fields, axis=3).max(axis=(1, 2))
            dt = np.diff(m[:n, 1]) if m.size and len(m) > 2 else np.zeros(0)
            hz = 1e9 / float(np.median(dt)) if dt.size and np.median(dt) > 0 else float("nan")
            clips.append((label, fields, peak, folder, hz, "pi"))
    return clips


def windows(label, fields, peak, lo, hi, stride=1):
    """Stacks of STACK consecutive fields, oldest first, ending at frames lo..hi-1.

    Channel order matches BosEngine::process: dx, dy of the oldest field, then the next, ...
    Built with one copy (sliding window view), not one small array per window.
    """
    lo = max(lo, STACK - 1)
    if hi <= lo:
        return np.zeros((0, GRID, GRID, 2 * STACK), np.float32), np.zeros(0, np.int64)
    v = np.lib.stride_tricks.sliding_window_view(fields[lo - STACK + 1: hi], STACK, axis=0)  # (n, G, G, 2, S)
    v = v[::stride].transpose(0, 1, 2, 4, 3)                                                  # (n, G, G, S, 2)
    X = np.ascontiguousarray(v, dtype=np.float32).reshape(-1, GRID, GRID, 2 * STACK)
    y = np.full(len(X), LABELS.index(label), np.int64)
    if label in ACTIVE_PX:
        y[peak[lo:hi][::stride] < ACTIVE_PX[label]] = 0
    return X, y


def prep(X):
    np.clip(X, -CLIP_PX, CLIP_PX, out=X)        # in place: the data set is the big allocation
    X *= SCALE
    return X


def data_report(clips):
    print("\ndata report")
    for label, fields, peak, src, hz, kind in clips:
        quiet = 100.0 * np.mean(peak < ACTIVE_PX[label]) if label in ACTIVE_PX else 0.0
        print(f"  {os.path.basename(src):34s} {kind:3s} {label:6s} {len(fields):5d} fields {hz:5.1f} Hz  "
              f"peak p50 {np.percentile(peak, 50):.3f} p95 {np.percentile(peak, 95):.3f}  relabeled clear {quiet:3.0f}%")
        if np.isfinite(hz) and abs(hz - PROC_HZ) > 0.15 * PROC_HZ:
            fix = ("the Pi's rate check is skipping frames (fix for Jaime in CHANGES.md)" if kind == "pi"
                   else "record with the updated bos_proto.py")
            print(f"    WARNING: {hz:.1f} Hz, not {PROC_HZ:.0f}: {fix}")
        if label in ACTIVE_PX and quiet > 85:
            print(f"    WARNING: almost every frame is below the {label} threshold. Plume too weak, or threshold too high?")
    for kind in ("mac", "pi"):
        clear_peaks = [p for l, _, p, _, _, k in clips if k == kind and l == "clear"]
        if not clear_peaks:
            continue
        floor = float(np.percentile(np.concatenate(clear_peaks), 99))
        print(f"  noise floor ({kind}, p99 of peak on clear clips): {floor:.3f} px")
        for label, thr in ACTIVE_PX.items():
            if thr < 1.2 * floor:
                print(f"    WARNING: ACTIVE_PX['{label}'] = {thr} is at or below the {kind} noise floor, so pure noise "
                      f"in {label} clips stays labeled {label}. Try about {1.3 * floor:.2f}.")
    print()


def make_batches(X, y, sample_w, batch, seed):
    """Shuffled batches with augmentation done on the fly, so the data set is never copied 3x."""
    import keras

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(X))

        def __len__(self):
            return int(np.ceil(len(X) / batch))

        def __getitem__(self, i):
            idx = np.sort(self.order[i * batch:(i + 1) * batch])
            xb, yb = X[idx], y[idx]                    # fancy indexing copies, X stays untouched
            # Mirror left-right and negate dx. Never flip up-down: that turns sinking gas into rising heat.
            flip = self.rng.random(len(idx)) < 0.5
            if flip.any():
                xf = xb[flip][:, :, ::-1, :].copy()
                xf[..., 0::2] *= -1.0
                xb[flip] = xf
            xb += 0.02 * self.rng.standard_normal(xb.shape, dtype=np.float32)
            return xb, yb, sample_w[yb]

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(X))

    return Batches()


def build_model():
    import tensorflow as tf
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input((GRID, GRID, 2 * STACK)),
        L.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
        L.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
        L.Conv2D(48, 3, strides=2, padding="same", activation="relu"),
        L.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dense(32, activation="relu"),
        L.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def to_tflite(model, workdir):
    """Plain float32 TFLite with builtin ops only, tried three ways for different TF versions."""
    import tensorflow as tf
    errors = []
    try:
        return tf.lite.TFLiteConverter.from_keras_model(model).convert()
    except Exception as e:                      # Keras 3 sometimes needs a SavedModel first
        errors.append(e)
    sm = os.path.join(workdir, "saved_model")
    try:
        model.export(sm)
    except Exception as e:
        errors.append(e)
        tf.saved_model.save(model, sm)
    try:
        return tf.lite.TFLiteConverter.from_saved_model(sm).convert()
    except Exception as e:
        errors.append(e)
    raise RuntimeError(f"TFLite conversion failed: {errors}")


def op_versions(tfl):
    """The ops and op versions the Pi's TFLite must support. Best effort: needs TF's schema module."""
    try:
        from tensorflow.lite.python import schema_py_generated as fb
    except Exception:
        return "unknown (no schema module in this TensorFlow)"
    m = fb.Model.GetRootAsModel(tfl, 0)
    names = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}
    ops = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        ops.append(f"{names.get(code, code)} v{oc.Version()}")
    return ", ".join(ops)


def interpreter(tfl):
    try:
        from ai_edge_litert.interpreter import Interpreter          # newer home of the interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_content=tfl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["recordings"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth training window (2 halves memory)")
    ap.add_argument("--out", default="export")
    args = ap.parse_args()

    clips = load_clips(args.data)
    if not clips:
        raise SystemExit("No data found. Record with bos_proto.py or copy Pi dumps here.")
    data_report(clips)

    # Split every clip in time: first 80% trains, the rest validates (with a gap, so no window
    # overlaps both sides). Splitting by frame at random would leak nearly identical neighbours.
    tr, va = [], []
    for label, fields, peak, src, hz, kind in clips:
        n = len(fields)
        cut = int(n * 0.8)
        tr.append(windows(label, fields, peak, 0, cut, args.stride))
        va.append(windows(label, fields, peak, cut + STACK, n))
    del clips
    Xtr, ytr = prep(np.concatenate([a for a, _ in tr])), np.concatenate([b for _, b in tr])
    del tr
    Xva, yva = prep(np.concatenate([a for a, _ in va])), np.concatenate([b for _, b in va])
    del va
    print("train counts:", {LABELS[i]: int((ytr == i).sum()) for i in range(len(LABELS))})
    print("val counts:  ", {LABELS[i]: int((yva == i).sum()) for i in range(len(LABELS))})
    print(f"training set {Xtr.nbytes / 1e9:.2f} GB")

    counts = np.bincount(ytr, minlength=len(LABELS)).astype(np.float64)
    sample_w = np.array([counts.sum() / (len(LABELS) * c) if c > 0 else 0.0 for c in counts], np.float32)

    model = build_model()
    model.fit(make_batches(Xtr, ytr, sample_w, 64, seed=0),
              validation_data=(Xva, yva) if len(Xva) else None, epochs=args.epochs, verbose=2)

    if len(Xva):
        pred = model.predict(Xva, verbose=0, batch_size=256).argmax(1)
        cm = np.zeros((len(LABELS), len(LABELS)), int)
        for t, p in zip(yva, pred):
            cm[t, p] += 1
        print("\nconfusion matrix (rows = truth, columns = predicted):")
        print("        " + " ".join(f"{l:>7s}" for l in LABELS))
        for i, l in enumerate(LABELS):
            print(f"{l:>7s} " + " ".join(f"{v:7d}" for v in cm[i]))
        gas_recall = cm[1, 1] / max(1, cm[1].sum())
        false_gas = (cm[:, 1].sum() - cm[1, 1]) / max(1, cm.sum() - cm[1].sum())
        print(f"gas recall {gas_recall:.1%}, false gas rate on everything else {false_gas:.1%}")

    os.makedirs(args.out, exist_ok=True)
    tfl = to_tflite(model, args.out)
    with open(os.path.join(args.out, "model.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(LABELS, f)
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump({"input": [1, GRID, GRID, 2 * STACK], "dtype": "float32",
                   "channels": "dx, dy for each field, oldest first",
                   "clip_px": CLIP_PX, "scale": SCALE, "proc_hz": PROC_HZ, "labels": LABELS,
                   "active_px": ACTIVE_PX, "ops": op_versions(tfl)}, f, indent=2)

    # The Pi will run exactly this file, so check it with the TFLite interpreter here.
    interp = interpreter(tfl)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    sample = Xva[:1] if len(Xva) else Xtr[:1]
    interp.set_tensor(inp["index"], sample)
    interp.invoke()
    print(f"\nexported {args.out}/model.tflite ({len(tfl) / 1024:.0f} KB), input {inp['shape'].tolist()}, "
          f"output {out['shape'].tolist()}, sample probs {np.round(interp.get_tensor(out['index'])[0], 3).tolist()}")
    print(f"ops the Pi needs: {op_versions(tfl)}")


if __name__ == "__main__":
    main()
