#!/usr/bin/env python3
"""train.py: trains the Mirage classifier and exports model.tflite.        owner: Gio

Data it reads (any mix, searched recursively):
  *.npz                 from bos_proto.py (fields, peak, label)
  <label>_<anything>/   Pi dumps from BosEngine::startDump: fields.f32 + meta.csv,
                        where the folder name starts with clear, gas, heat or motion
Writes export/model.tflite, export/labels.json, export/model_info.json.

  pip install tensorflow numpy
  python train.py --data recordings pi_dumps
"""
import argparse
import glob
import json
import os

import numpy as np

GRID, STACK, CLIP_PX, SCALE, PROC_HZ = 64, 4, 4.0, 0.5, 10.0   # must match BosParams
LABELS = ["clear", "gas", "heat", "motion"]                    # must match enum mirage_class
# Frames inside an active clip that barely move are really "clear" (between lighter puffs, say).
ACTIVE_PX = {"gas": 0.12, "heat": 0.04, "motion": 0.10}


def load_clips(roots):
    clips = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
            d = np.load(f, allow_pickle=False)
            clips.append((str(d["label"]), d["fields"].astype(np.float32), d["peak"].astype(np.float32), f))
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
            clips.append((label, fields, peak, folder))
    return clips


def windows(label, fields, peak, lo, hi):
    """Stacks of STACK consecutive fields, oldest first, ending at frames lo..hi-1."""
    X, y = [], []
    li = LABELS.index(label)
    for i in range(max(lo, STACK - 1), hi):
        X.append(np.concatenate(list(fields[i - STACK + 1: i + 1]), axis=-1))   # (G, G, 2*STACK)
        y.append(0 if (label in ACTIVE_PX and peak[i] < ACTIVE_PX[label]) else li)
    return X, y


def prep(X):
    return (np.clip(X, -CLIP_PX, CLIP_PX) * SCALE).astype(np.float32)


def augment(X, y, rng):
    # Mirror left-right and negate dx. Never flip up-down: that turns sinking gas into rising heat.
    Xf = X[:, :, ::-1, :].copy()
    Xf[..., 0::2] *= -1.0
    Xn = (X + rng.normal(0.0, 0.02, X.shape)).astype(np.float32)
    return np.concatenate([X, Xf, Xn]), np.concatenate([y, y, y])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["recordings"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="export")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    clips = load_clips(args.data)
    if not clips:
        raise SystemExit("No data found. Record with bos_proto.py or copy Pi dumps here.")

    # Split every clip in time: first 80% trains, the rest validates (with a gap, so no window
    # overlaps both sides). Splitting by frame at random would leak nearly identical neighbours.
    Xtr, ytr, Xva, yva = [], [], [], []
    for label, fields, peak, src in clips:
        n = len(fields)
        cut = int(n * 0.8)
        a, b = windows(label, fields, peak, 0, cut)
        c, d = windows(label, fields, peak, cut + STACK, n)
        Xtr += a; ytr += b; Xva += c; yva += d
        print(f"{os.path.basename(src):40s} {label:7s} {n:5d} fields")
    Xtr, ytr = prep(np.asarray(Xtr)), np.asarray(ytr)
    Xva, yva = prep(np.asarray(Xva)), np.asarray(yva)
    print("train counts:", {LABELS[i]: int((ytr == i).sum()) for i in range(len(LABELS))})
    print("val counts:  ", {LABELS[i]: int((yva == i).sum()) for i in range(len(LABELS))})

    Xtr, ytr = augment(Xtr, ytr, rng)
    counts = np.bincount(ytr, minlength=len(LABELS)).astype(np.float64)
    weights = {i: float(counts.sum() / (len(LABELS) * c)) for i, c in enumerate(counts) if c > 0}

    model = build_model()
    model.fit(Xtr, ytr, validation_data=(Xva, yva) if len(Xva) else None,
              epochs=args.epochs, batch_size=64, class_weight=weights, verbose=2)

    if len(Xva):
        pred = model.predict(Xva, verbose=0).argmax(1)
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
                   "clip_px": CLIP_PX, "scale": SCALE, "proc_hz": PROC_HZ, "labels": LABELS}, f, indent=2)

    # The Pi will run exactly this file, so check it with the TFLite interpreter here.
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    sample = Xva[:1] if len(Xva) else Xtr[:1]
    interp.set_tensor(inp["index"], sample)
    interp.invoke()
    print(f"\nexported {args.out}/model.tflite ({len(tfl) / 1024:.0f} KB), input {inp['shape'].tolist()}, "
          f"output {out['shape'].tolist()}, sample probs {np.round(interp.get_tensor(out['index'])[0], 3).tolist()}")


if __name__ == "__main__":
    main()
