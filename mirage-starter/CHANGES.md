# Changes to the starter kit (Gio, hour 0)

All of these were run end to end on synthetic dot-screen videos with TensorFlow 2.21 / Keras 3.15 /
OpenCV 4.13 before handing over. No contract changed: same model input and output, same constants,
same telemetry JSON.

## mac/ (Gio)

- **train.py**: the old version built every training window as its own array and then tripled the
  set for augmentation, in float64 for the noise. At the recording protocol's size (12 Pi clips of
  600 fields plus Mac clips) it was killed for running out of memory on a 4 GB machine; the new one
  peaks at 2.5 GB on the same data. Windows and labels are bit-identical to before (checked);
  augmentation (left-right mirror with dx negated, small noise) now happens per batch.
  It also prints a data report before training: processing rate per clip, peak shift, how many quiet
  frames were relabeled clear, and a warning when an `ACTIVE_PX` threshold sits under the noise floor
  of the clear clips. `--stride 2` halves memory if needed. `model_info.json` now lists the ops.
- **bos_proto.py**: fixed processing rate (see below). Docstring: videos must start with 2 s of
  nothing in view, because the first frame becomes the reference.
- **dashboard.py**: survives a malformed packet; status text no longer runs off the right edge.
- **fake_pi.py** (new): sends the supervisor's exact JSON plus shift-map thumbnails on a 30 s loop
  (boot, leak trip, deadline trip, heat that must not trip), or replays real `.npz` recordings.
  The dashboard can be finished without the Pi.
- **dots.html** (new): the dot background. Open it on the screen behind the plume.
- **export_v0/model_v0_PLUMBING_ONLY.tflite**: trained on synthetic video only. It proves the file
  loads on the Pi and matches the I/O contract (float32 1x64x64x8 in, 1x4 out; ops CONV_2D, MEAN,
  FULLY_CONNECTED, SOFTMAX, all version 1, min runtime 1.6). Its verdicts mean nothing on real data.

## For Jaime: the processing rate is not 10 Hz on real cameras

`BosEngine::process` compares each frame with the last processed one. With a 30 fps camera, three
frames later is 99.99 ms if the sensor runs even 0.03% fast, which is under 100 ms, so it waits for a
fourth frame: 7.5 Hz. With 2 ms of timestamp jitter it averages 8.4 Hz. Simulated over 60 s of
frames, the fix below holds 10.0 Hz for 24, 25, 29.97, 30, 30.01 and 60 fps cameras.

`bos_engine.hpp`: rename the member

```cpp
uint64_t next_proc_ns_ = 0;          // was: uint64_t last_proc_ns_ = 0;
```

`bos_engine.cpp`, top of `process()`: replace the rate check and `last_proc_ns_ = frame_ns;` with

```cpp
const uint64_t period_ns = static_cast<uint64_t>(1e9 / p_.proc_hz);
if (!want_ref_ && next_proc_ns_ && frame_ns + period_ns / 5 < next_proc_ns_) {
    return false;                                   // keep the processing rate fixed
}
next_proc_ns_ = (next_proc_ns_ && frame_ns < next_proc_ns_ + period_ns) ? next_proc_ns_ + period_ns
                                                                        : frame_ns + period_ns;
```

Quick check after it’s in: in any dump’s `meta.csv`, consecutive `frame_ns` values should differ by
about 100,000,000. `train.py` warns per clip when they don’t.
