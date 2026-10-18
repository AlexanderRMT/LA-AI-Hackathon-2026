#!/usr/bin/env python3
"""test_train_tuned.py: train_tuned.py must build exactly what train.py builds.   owner: Gio

With no presses.txt and the synthetic taps off, the windows, labels, class weights and augmented
batches must equal train.py's, value for value. Also checks the press labels, the synthetic taps,
the gas bias (it must move the softmax exactly like a shifted gas logit), and the supervisor rule
used to score clips (3 gas verdicts in a row). Synthetic clips in a temp folder; no recordings.

  python mac/test_train_tuned.py        (or: pytest mac/test_train_tuned.py)
"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import train_tuned as TT  # noqa: E402

G = T.GRID


def write_clip(root, name, peaks, seed, presses=None):
    """A Pi dump folder (fields.f32 + meta.csv) whose meta peaks are the given values."""
    rng = np.random.default_rng(seed)
    d = Path(root) / name
    d.mkdir(parents=True)
    fields = (rng.standard_normal((len(peaks), G, G, 2)) * 0.05).astype(np.float32)
    fields[:, 5, 7, 0] = peaks                    # a few strong cells so clipping is exercised too
    fields[::7, 9, 9, 1] = 6.0
    fields.tofile(d / "fields.f32")
    with open(d / "meta.csv", "w") as f:
        f.write("idx,frame_ns,peak_px,bump_px\n")
        for i, p in enumerate(peaks):
            f.write(f"{i},{1_000_000_000 + i * 100_000_000},{p:.4f},0.0100\n")
    if presses is not None:
        np.savetxt(d / "presses.txt", presses)
    return d


def dataset(root, with_presses=False):
    rng = np.random.default_rng(1)
    n = 120
    write_clip(root, "clear_01", rng.uniform(0.05, 0.2, n), 1)
    write_clip(root, "gas_lighter_01", rng.uniform(0.0, 0.3, n), 2, [2.0, 7.0] if with_presses else None)
    write_clip(root, "heat_mug_01", rng.uniform(0.0, 3.0, n), 3)
    write_clip(root, "motion_wave_01", rng.uniform(0.0, 0.4, n), 4)
    return root


def args_off(**over):
    a = dict(seed=0, stride=1, epochs=1, zoom=0.0, zoom_px=(0.03, 0.6), gas_bias=0.0)
    a.update(over)
    return types.SimpleNamespace(**a)


def test_windows_and_labels_match_train_py():
    with tempfile.TemporaryDirectory() as root:
        dataset(root)
        X, end, cid, meta, clips = TT.load([root])
        y, w = TT.all_labels(end, cid, meta, dict(T.ACTIVE_PX))
        assert np.all(w == 1.0)
        for k, (label, fields, peak, src, hz, kind) in enumerate(clips):
            Xt, yt = T.windows(label, fields, peak, 0, len(fields))
            Xt = T.prep(Xt)
            sel = cid == k
            assert np.array_equal(X[sel], Xt), f"windows differ for {src}"
            assert np.array_equal(y[sel], yt), f"labels differ for {src}"


def test_batches_match_train_py_when_extras_are_off():
    with tempfile.TemporaryDirectory() as root:
        dataset(root)
        X, end, cid, meta, clips = TT.load([root])
        y, w = TT.all_labels(end, cid, meta, dict(T.ACTIVE_PX))
        counts = np.bincount(y, minlength=len(T.LABELS)).astype(np.float64)
        cw = np.array([counts.sum() / (len(T.LABELS) * c) if c > 0 else 0.0 for c in counts], np.float32)
        theirs = T.make_batches(X, y, cw, 64, seed=0)
        ours = TT.batches(X, np.arange(len(X)), y, w, cw, args_off(), seed=0)
        assert len(theirs) == len(ours)
        for i in range(len(ours)):
            for a, b in zip(theirs[i], ours[i]):
                assert np.array_equal(a, b), f"batch {i} differs"


def test_press_labels():
    with tempfile.TemporaryDirectory() as root:
        dataset(root, with_presses=True)
        X, end, cid, meta, clips = TT.load([root])
        m = next(m for m in meta if m["label"] == "gas")
        assert list(m["presses"]) == [2.0, 7.0]
        m = dict(m, peak=np.full(len(m["peak"]), 0.2, np.float32))      # strong everywhere
        y, w = TT.frame_labels(m, {"gas": 0.12})
        t = np.arange(len(y)) / T.PROC_HZ
        on = ((t >= 2.5) & (t < 5.5)) | ((t >= 7.5) & (t < 10.5))
        assert np.all(y[on] == 1) and np.all(w[on] == 1)
        assert np.all(y[~on] == 0) and np.all(w[~on] == 0)              # strong but no press: left out
        m["peak"][:] = 0.05                                              # weak everywhere: clear, kept
        y, w = TT.frame_labels(m, {"gas": 0.12})
        assert np.all(y == 0) and np.all(w == 1)
        m["peak"][30] = 5.0                                              # the lighter in the box: left out
        y, w = TT.frame_labels(m, {"gas": 0.12}, solid_px=3.0)
        assert w[30] == 0 and np.sum(w == 0) == 1


def test_synthetic_taps():
    rng = np.random.default_rng(0)
    field, size = TT.zoom_nuisance(rng, 500, 0.03, 0.6)
    px = field / T.SCALE
    assert field.shape == (500, G, G, 2 * T.STACK) and field.dtype == np.float32
    assert np.abs(np.median(px[..., 0::2], axis=(1, 2))).max() < 1e-6   # bump removal leaves no shift
    edge = np.abs(px).max(axis=(1, 2, 3))
    assert np.all(edge <= 1.4 * size + 1e-6) and np.all(edge >= 0.8 * size - 1e-6)    # corner: stretch 0.6-1.2 plus tilt 0.2
    assert 0.03 <= size.min() and size.max() <= 0.6
    # every window gets a tap, every label stays what it was, and the data set itself is untouched
    X = np.zeros((64, G, G, 2 * T.STACK), np.float32)
    y = np.arange(64) % len(T.LABELS)
    cw = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    b = TT.batches(X, np.arange(64), y, np.ones(64, np.float32), cw, args_off(zoom=1.0, zoom_px=(0.3, 0.6)), seed=0)
    xb, yb, wb = b[0]
    assert np.array_equal(yb, y) and np.array_equal(wb, cw[y])
    assert np.all(np.abs(xb).max(axis=(1, 2, 3)) > 0.1) and not X.any()


def test_gas_bias_shifts_the_gas_logit():
    model = T.build_model()
    x = np.random.default_rng(0).standard_normal((8, G, G, 2 * T.STACK)).astype(np.float32)
    p0 = model.predict(x, verbose=0).astype(np.float64)
    TT.add_gas_bias(model, -1.25)
    p1 = model.predict(x, verbose=0).astype(np.float64)
    want = p0.copy()
    want[:, 1] *= np.exp(-1.25)
    want /= want.sum(axis=1, keepdims=True)
    assert np.allclose(p1, want, atol=1e-5)


def test_supervisor_rule():
    def probs(gas_pattern):
        p = np.tile([0.97, 0.01, 0.01, 0.01], (len(gas_pattern), 1))
        p[np.array(gas_pattern, bool)] = [0.05, 0.9, 0.03, 0.02]
        return p
    m = {"name": "c", "label": "clear", "presses": None}
    t = np.arange(12) / 10.0
    assert TT.score_clip(m, t, probs([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]))["trips"] == 0
    assert TT.score_clip(m, t, probs([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]))["trips"] == 2
    p = np.tile([0.3, 0.4, 0.2, 0.1], (3, 1))                           # top class gas below 0.60 still counts
    assert TT.score_clip(m, t[:3], p)["trips"] == 1
    p = np.tile([0.35, 0.3, 0.05, 0.3], (3, 1))                         # not top and below 0.60: not gas
    assert TT.score_clip(m, t[:3], p)["trips"] == 0
    g = {"name": "g", "label": "gas", "presses": np.array([0.2, 1.0])}
    t = np.arange(16) / 10.0
    r = TT.score_clip(g, t, probs([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert (r["presses_hit"], r["before_first_press"]) == (1, 0) and abs(r["latency_s"] - 0.3) < 1e-9


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
