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

## mac/ web console (Gio)

- **web_dashboard.py** and **web/** (new): the dashboard in a browser. `python mac/web_dashboard.py`
  opens http://127.0.0.1:8750. It reads UDP 5005 and 5006 exactly like dashboard.py (same buffer
  sizes, 1 s stale rule, valve shown open only while fresh, class names, 120-update history, p99 over
  the last 600 latencies) and only listens; it never sends anything to the Pi. dashboard.py is
  unchanged and stays the fallback. Only one of the two can own the ports at a time, and the console
  says so if dashboard.py is already running.
- What it adds is context from data we already receive: which machine is sending (two senders at once
  get a warning), a banner when the data comes from this Mac (fake_pi.py), stale or missing images,
  how old the last verdict is, and the trip rules read from common/mirage_msgs.h. It can start and
  stop fake_pi.py from the page, only from this Mac and never while real telemetry is arriving.
- **test_web_dashboard.py** (new): replays the same packets through dashboard.py's real loop and the
  console and checks that every displayed line and history bar match: `python mac/test_web_dashboard.py`.
- Not fixed, for the record: dashboard.py raises if a packet is valid JSON but not an object (`42`),
  or has a non-numeric `p_gas`, `cls` or `lat_ms`. The supervisor never sends those; the console skips them.

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

## Web console: camera view and new reference frame (Alex)

- While telemetry is live from the Pi, the image panel shows **Camera**: mirage_vision's own view of
  the dot screen region with a margin around it (MJPEG from `http://<pi>:8080/region`, shift map
  glowing over the region, box green on clear and red on gas, status bar underneath). Without a live
  Pi (nothing yet, or fake_pi.py) or if that stream can't be reached, it falls back to the shift-map
  thumbnail as before. `--pi-view-port` if mirage_vision runs with a different `-p`.
- **New reference frame** button in that panel: POST `/api/reference`, which the console forwards to
  the Pi's mirage_vision (same as `slay -s SIGUSR1 mirage_vision`). Same guard as the simulator
  controls (loopback, `X-Mirage-Control`, same origin). It's the one thing the console sends the Pi.
- `.stage img` is now pinned to the stage: `height: 100%` doesn't resolve in a grid item, so a tall
  image overflowed and was clipped.
- Shift-map thumbnails never arrived before: QNX refuses UDP datagrams over 9216 bytes and the JPEGs
  were bigger. mirage_vision now steps the quality down until they fit (~7 KB).

## mac/ model v1 (Gio)

- **train_tuned.py** (new): trains model v1 on the Pi dumps. It imports train.py's windows,
  preprocessing, model and TFLite export, so the contract holds (float32 1x64x64x8 in, 1x4 softmax
  out; CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX, all v1; 198 KB, same size and speed as v0).
  train.py is unchanged. `python mac/train_tuned.py --data pi_dumps --check` (held-out check, then
  train on every clip and write export/), about 8 min on an M1.
- `--check` scores every clip with a model trained without it, through the supervisor's own rule
  (top class gas or p_gas >= 0.60, 3 in a row close the valve). On the 15 clips of 2026-10-18:
  **47 of 50 lighter presses closed the valve, median 1.0 s after the press started; 2 false trips
  on 10 clear, heat and motion clips (both in one hand-waving clip); 0 on clear, the mug and table
  taps; 0.37% of their frames read as gas.** train.py's settings, in the same kind of held-out check,
  gave 45 of 50 presses and 6 false trips (2.8% of frames). gas_lighter_04 (nearer the screen) is the
  weak one: 7 of 10.
- What changed, each kept only because the held-out check improved:
  - Gas labels from the press timing: `presses.txt` in each gas clip folder lists the press start
    times (5, 10, ... 50 s). A frame is gas while gas can be in the box (0.5 to 3.5 s after a press
    starts) and the shift shows; strong frames between presses, and frames with something solid in
    the box (over 3 px: the lighter dipped in), are left out of training.
  - Synthetic table taps on 30% of training windows. After median bump removal a tap leaves a
    whole-box zoom (an affine fit explains 96% of what's left; gas 6%), so the model learns that a
    smooth zoom is not gas. With them the unseen tap clip had no false trips in all three runs
    (0 to 19 without, depending on the random seed).
  - Gas bias -1.5 on the last layer: fewer gas verdicts on borderline frames at no cost in presses.
- Tried and dropped: mixing clear-clip noise into every window, random signal strength, the data
  report's higher thresholds, and labeling strong synthetic taps as motion (the model then called
  people walking past gas).
- **test_train_tuned.py** (new): with no `presses.txt` and the taps off, windows, labels, class
  weights and augmented batches equal train.py's value for value; also checks the press labels, the
  taps, the gas bias and the trip rule. `python mac/test_train_tuned.py`
- For the record, not changed: the 5 "trips before the first press" in the held-out check are at the
  bottom rows of the box, sometimes before anyone was in view. Butane is twice as dense as air and the
  gas clips were recorded back to back, so these are probably real gas pooled from earlier presses.
  An affine fit instead of the median in bump removal would take tap zoom out before the model sees
  it; that is contract 3 (BosParams and mac/*.py), so only with all three of us.
- On the Pi the model is `/data/home/qnxuser/model.tflite`. Add `-m /data/home/qnxuser/model.tflite`
  to the mirage_vision command and look for `[bos] model loaded:` in its output.

## vision/: a model that doesn't load no longer falls back silently (Gio, for Jaime)

- `mirage_hook.cpp` ignored what `loadModel()` returned. With a wrong `-m` path, a corrupt file or
  an op the Pi's TFLite lacks, mirage_vision printed one `[bos]` line and kept running on the physics
  rule, which calls the hot mug gas (its shift is 1.3 to 1.9 px; the rule trips at 0.30). Now
  mirage_vision checks the model before it opens the camera and won't start without it (exit 1 with
  the reason; under `run_mirage.sh` it keeps retrying while the supervisor stays closed). If the
  load still fails later, no verdicts are sent, so the valve closes on the supervisor's deadline.
  Without `-m` nothing changed.
- Tested on the Pi from a separate build: a missing file and a junk file both refuse to start with
  the reason; `export/model.tflite` loads in TFLite 2.21 (XNNPACK on the CPU).
