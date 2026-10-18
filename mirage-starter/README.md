# Mirage starter kit

Mirage sees gas leaks with a camera and a dot screen (background-oriented schlieren), classifies
the plume with a TFLite model on a Raspberry Pi 5 running QNX, and lets a top-priority supervisor
close a normally-closed valve. Explainer and live simulation:
https://claude.ai/artifact/Lzb7oH76Uhe4io8HfREiKL

```
                ┌───────────────────────── Raspberry Pi 5, QNX 8 ──────────────────────────┐
 dot screen ──▶ │ Camera Module 3 → Sensor Framework → mirage_vision (priority 20)         │
 (a laptop)     │   BOS engine: DIS flow, bump removal, 64×64 field → TFLite model         │
                │        │ one verdict every 100 ms over QNX message passing                │
                │        ▼                                                                  │
                │   supervisor (priority 50, SCHED_FIFO) → relay → normally-closed valve   │
                │        LEDs, buzzer, reset button                                         │
                └────────┬──────────────────────────────────────────────────────────────────┘
                         │ UDP 5005 telemetry (JSON), UDP 5006 shift-map thumbnails (JPEG)
                         ▼
                   Gio’s Mac: mac/dashboard.py
```

## Who owns what

|                | Jaime (QNX)                                   | Alex (QNX)                                         | Gio (Mac)                                         |
|----------------|-----------------------------------------------|----------------------------------------------------|---------------------------------------------------|
| Builds         | `vision/`: camera → BOS → model → verdicts    | `supervisor/`, `tools/`, wiring, the rigid rig     | `mac/`: physics proof, data, model, dashboard     |
| First hour     | QNX booted on the Pi, workshop camera demo running with the Camera Module 3 | Supervisor and mock_vision talking on a QNX VM (NO_GPIO build) | A lighter plume showing up in `bos_proto.py` on the Mac camera |
| Hands off      | Field dumps to Gio, verdicts to the supervisor | Telemetry to Gio, valve state to everyone          | `model.tflite` to Jaime                           |
| Also owns      | ROI and camera settings                       | Chaos demo, latency numbers                        | Devpost, demo video, props, the dot-screen device |
| Pitch section  | The physics and the founder story             | Why QNX, and the live chaos demo                   | The AI, the data, and the impact                  |

Gio doesn’t need SDP. Everything in `mac/` runs on macOS. If he needs the Pi, he can SSH in
(`ssh qnxuser@qnxpi.local`) and build on the target, since the current QNX images ship a self-hosted
developer desktop.

## The contracts (change these only with all three of you)

1. **Verdicts:** `common/mirage_msgs.h`. The vision process sends `mirage_verdict_t` with
   `MsgSend()` to the name `mirage_supervisor` and gets `mirage_status_t` back.
2. **Model:** input `float32 [1, 64, 64, 8]`, which is (dx, dy) for the last 4 fields, oldest first,
   clipped to ±4 px and multiplied by 0.5. Output `float32 [1, 4]` softmax over
   `clear, gas, heat, motion`. Fields are processed at 10 Hz.
3. **Processing constants:** DIS optical flow `PRESET_MEDIUM`, median bump removal, `INTER_AREA`
   resize to 64×64, EMA 0.5. They live in `BosParams` (C++) and at the top of `mac/*.py`.
4. **Training data:** Pi dumps are folders named after the label (`gas_lighter_01/`) holding
   `fields.f32` (N×64×64×2 float32) and `meta.csv` (`idx,frame_ns,peak_px,bump_px`).
   Mac recordings are `.npz` files from `bos_proto.py`.
5. **Telemetry:** the supervisor sends JSON to `<mac ip>:5005` at 10 Hz. The vision hook sends
   JPEG thumbnails to `<mac ip>:5006`.
6. **Pins:** listed under Wiring below.

## Plan and checkpoints

Work backward from the official “hacking ends” time on live.lahacks.com. Hours below count from
when you start.

| When            | Jaime                                       | Alex                                        | Gio                                       |
|-----------------|---------------------------------------------|---------------------------------------------|-------------------------------------------|
| Hour 0 to 1     | Boot QNX, clone `qnx/QNX-AI-workshop`, camera running | Supervisor + mock_vision on the VM           | Kill test in `bos_proto.py`; get props    |
| **Checkpoint 1**| All three report: frames on the Pi, IPC working, plume visible on the Mac |  |  |
| Hour 1 to 3     | `mirage_hook.cpp` in the workshop app, physics-only rule, dumping fields | Supervisor on the Pi with real GPIO; wire relay, LEDs, buzzer, button; build the rig | Tune DIS on the Mac; dashboard against telemetry from mock_vision |
| **Checkpoint 2**| First real trip: lighter puff → Pi → supervisor → relay clicks |  |  |
| Hour 3 to 4     | Recording session, all three, on the final rig (protocol below) |  |  |
| Hour 4 to 7     | Load model v1, measure CPU time per frame   | Latency logging, run_mirage.sh, false-trip runs | Train v1, then v2 on Pi dumps; dashboard polish |
| **Checkpoint 3**| Model in the loop, dashboard live           |                                             |                                           |
| Until freeze    | Edge cases: people walking past, lighting, bumps | Chaos demo rehearsed                     | Devpost draft, slides if any              |
| **Freeze**      | Four hours before hacking ends: no new features. Record the video, submit, rehearse three times. |  |  |

## Recording protocol (Hour 3 to 4)

Set the rig once and never touch it again: camera on a rigid mount about 1 m from the screen,
the gas or heat source about halfway. Reset the reference after any bump.

| Folder prefix | What to do for 60 s per clip, at least 3 clips each                  |
|---------------|----------------------------------------------------------------------|
| `gas_`        | Unlit lighter: 1 s press, 3 s pause. Also keyboard duster.           |
| `heat_`       | Soldering iron from the hardware desk, hair dryer, hot mug           |
| `motion_`     | Hand waving in front, walking past, light table bumps                |
| `clear_`      | Nothing in front, normal room activity                               |

On the Pi: `mkdir -p /data/home/qnxuser/rec/gas_lighter_01`, start the vision app with that dump dir,
then on the Mac: `scp -r qnxuser@qnxpi.local:/data/home/qnxuser/rec ./pi_dumps`.
Quiet frames inside an active clip are relabeled clear automatically by `train.py`.

## Commands

**Alex, on the VM first**
```sh
qcc -Vgcc_ntox86_64 -O2 -DNO_GPIO -o supervisor supervisor/supervisor.c -lsocket
qcc -Vgcc_ntox86_64 -O2 -o mock_vision supervisor/mock_vision.c
./supervisor &          # boots CLOSED
./mock_vision           # c clear, g gas, s stall, x reset
```
Then for the Pi (run as root; if `<sys/rpi_gpio.h>` is missing on the host, copy it from the
QNX hardware samples repo’s `common/` folder):
```sh
qcc -Vgcc_ntoaarch64le -O2 -o supervisor supervisor/supervisor.c -lsocket
qcc -Vgcc_ntoaarch64le -O2 -o mirage_ctl tools/mirage_ctl.c
qcc -Vgcc_ntoaarch64le -O2 -o cpu_hog tools/cpu_hog.c
scp supervisor mirage_ctl cpu_hog tools/run_mirage.sh qnxuser@qnxpi.local:/data/home/qnxuser/
```
On the Pi, check `ls /dev/gpio` first. The supervisor drives outputs through `/dev/gpio/msg`.

**Jaime**
1. Clone `qnx/QNX-AI-workshop`, run its `configure.sh`, and get the hand demo showing the camera.
   On image 8.0.5 or newer the README says which camera unit to pass.
2. Add `vision/bos_engine.cpp` and `vision/mirage_hook.cpp` to its build. Call
   `mirage_init(roi, model_path, dump_dir, dashboard_ip)` once and `mirage_on_frame(frame, ts)` from
   the camera callback in place of the hand models.
3. Find the ROI by saving one frame and measuring where the dot screen sits.
4. Run it at priority 20 (`on -p 20 ./app`) or through `tools/run_mirage.sh`.
5. If `cv::DISOpticalFlow` is missing from the Pi’s OpenCV build, swap in
   `cv::calcOpticalFlowFarneback` and tell Gio so the Mac side matches.

**Gio**
```sh
pip install opencv-python numpy tensorflow
python mac/bos_proto.py                       # kill test with the Mac camera
python mac/bos_proto.py --video clip.mov      # or a phone video
python mac/train.py --data recordings pi_dumps
python mac/dashboard.py
scp export/model.tflite qnxuser@qnxpi.local:/data/home/qnxuser/
```
`train.py` prints a confusion matrix plus gas recall and false-gas rate. Put those numbers in the
pitch. The exported model uses only basic ops (CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX), all
version 1. If the Pi still rejects it, re-export with an older TensorFlow that matches the Pi’s
TFLite.

## Wiring (BCM GPIO numbers, physical pins in parentheses)

| Part              | Connection                                                              |
|-------------------|-------------------------------------------------------------------------|
| Relay module      | VCC to 5 V (2), GND (6), IN to GPIO17 (11). Valve or demo load on COM and NO, so it only gets power while the relay is energized |
| Green LED, open   | GPIO27 (13) through a resistor to GND (14)                              |
| Red LED, tripped  | GPIO22 (15) through a resistor to GND (20)                              |
| Buzzer module     | Signal GPIO23 (16), VCC 3.3 V (17), GND (25)                            |
| Reset button      | Signal GPIO24 (18), VCC **3.3 V (1), never 5 V**, GND (39)              |

If the relay clicks on when IN is low, set `RELAY_ACTIVE_HIGH 0` in `supervisor.c`. Prefer an
active-high module, because then a dead Pi can never hold the valve open.

## The chaos demo (about 60 s at the table)

1. Valve open, dashboard green.
2. Puff the lighter. The valve trips; the dashboard shows “first gas frame to valve” in ms.
3. Press reset while gas is still there: refused. Press again once it clears: accepted.
4. `./cpu_hog 15 4 10`: nothing happens, because vision outranks the hog.
5. `./cpu_hog 30 4 10`: vision starves, the supervisor fails closed within 250 ms. Your SSH session
   freezes for those 10 s; that’s expected.
6. `slay` the vision process: the supervisor trips on silence, `run_mirage.sh` restarts vision, and the
   valve stays closed until a person resets it.

## Answers to have ready

- **Why QNX?** Fixed priorities, a microkernel that isolates the vision process from the safety
  process, and message passing where the server inherits the sender’s priority. The deadline check
  runs off a priority-50 timer pulse, so a stalled or killed vision process can’t keep the valve open.
- **What if the supervisor dies?** It’s a few hundred lines with no dynamic allocation, at the top
  priority. In production the relay would sit behind a hardware watchdog that drops out unless the
  supervisor keeps toggling it.
- **False alarms from cooking?** Heat is its own class, three consecutive gas frames are required, and
  the confusion matrix shows the false-gas rate.
- **Methane?** We can’t bring natural gas indoors. It bends light more than air like butane does but
  rises, so the same physics applies.

## Devpost skeleton (Gio)

Inspiration (Aliso Canyon, gas line work, SF6 in substations) · What it does · How we built it
(BOS, QNX processes and priorities, TFLite) · Challenges (sub-pixel shifts, bumps, train/serve
skew) · Accomplishments (measured latency, chaos demo) · What’s next (hydrogen, natural backgrounds,
hardware watchdog).
