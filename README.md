# Mirage: seeing gas leaks with a camera

**Mirage spots an invisible gas leak on camera and shuts the valve in about a second.** It runs on a
Raspberry Pi 5 with the QNX real-time OS. A safety supervisor keeps the valve closed whenever it can't
confirm the air is clear.

Built in a weekend at LA Hacks 2026 by Jaime Rebollar, Alexander Magdaleno and Giovanni Rodriguez.

## Inspiration

Gas leaks are invisible. The 2015 Aliso Canyon blowout leaked methane for four months. Smaller leaks
in homes, gas lines and substations (SF6 in switchgear) go unnoticed until someone smells them, or
until something ignites. Point gas sensors only alarm once gas reaches them, and optical gas imaging
cameras cost tens of thousands of dollars.

Gas bends light slightly differently from air. Labs photograph airflow with a technique called
**background-oriented schlieren (BOS)**: put a random dot pattern behind the air you're watching, and
anything that bends light makes the dots appear to shift by a fraction of a pixel. We wanted to know
whether a $25 camera, a laptop screen showing dots, and a Raspberry Pi could do the same thing in
real time, and whether we could make it safe enough to control a valve.

## What it does

1. A **dot screen** sits behind the area being watched. The **Pi camera** watches it.
2. **Vision** compares each frame with a reference frame and measures how far every part of the dot
   pattern has shifted. A plume of gas shows up as a patch of sub-pixel shifts.
3. A small **TensorFlow Lite model** reads the last 0.4 s of that shift field and classifies it as
   *clear*, *gas*, *heat* (hot air from a mug or a soldering iron) or *motion* (people, hands, bumps).
4. A **safety supervisor** gets a verdict 10 times a second. After 3 gas verdicts in a row it
   **closes the valve**. If vision stalls, crashes or is starved of CPU for 250 ms, it **fails
   closed**. It only lets a person reopen the valve once it has seen 5 clear verdicts in a row.
5. A **six-LED panel** on the Pi and a **live web dashboard** show what's happening: the camera
   with the gas glowing over the dots, the valve state, the latest verdict and a gas-probability chart.

## How we built it

```
 dot screen ──▶ Pi Camera Module 3 ──▶ mirage_vision (QNX, priority 20)
                                        DIS optical flow → bump removal → 64×64 shift field → TFLite
                                               │ verdict every 100 ms (QNX message passing)
                                               ▼
                                        supervisor (QNX, priority 50, SCHED_FIFO)
                                               ├─▶ valve (relay / valve LED) + six-LED panel
                                               └─▶ telemetry over UDP ──▶ web dashboard
```

- **QNX 8 on a Raspberry Pi 5.** The camera comes in through QNX's Sensor Framework. The vision
  process and the safety supervisor are separate processes with fixed priorities. The supervisor's
  deadline check runs on its own priority-50 timer, so a stalled or killed vision process can't keep
  the valve open.
- **Vision (C++, OpenCV, TensorFlow Lite).** DIS optical flow on the dot-screen region, median
  subtraction to remove whole-image bumps, then a 64×64 field into the model. About 55 ms per frame
  on the Pi, at a steady 10 verdicts per second.
- **Model.** Trained on recordings made on the final rig: lighter presses, hot mugs, soldering iron,
  hands waving, people walking past, table taps. It uses only basic TFLite ops, so it runs on the
  Pi's TFLite build.
- **Supervisor (C).** About 450 lines with no dynamic allocation. Drives the valve output and LEDs
  through QNX's GPIO resource manager.
- **Dashboard.** Python plus a web page: the Pi's live camera view, the valve, the latest verdict
  and history, with *New reference frame* and *Open valve* buttons. Opening the valve still goes
  through the supervisor's rules.

## Results

Measured on the rig, with the model scoring recordings it wasn't trained on:

| | |
|---|---|
| Lighter presses that closed the valve | **47 of 50** |
| Time from press to valve closed (median) | **1.0 s** |
| False trips on clear, heat and motion clips | **2**, both from one hand-waving clip |
| Hot mug, table taps, empty scene | **0** false trips |
| Camera frame to supervisor (99th percentile) | **about 55 ms** |
| Vision silent → valve closed | **≤ 250 ms** |

**The chaos demo:** load the CPU with a lower-priority process and nothing happens, because vision
outranks it. Load it above vision and the supervisor fails closed within 250 ms. Kill vision and the
valve closes and stays closed until a person reopens it.

## Challenges

- **Sub-pixel signals.** A butane plume shifts the dots by only 0.2 to 0.5 px. Every bump to the table,
  change in light or sleeping laptop screen is larger than that, so the reference frame and a rigid
  rig matter as much as the code.
- **Gas versus everything else.** A simple "big shift means gas" rule also fires on hands, hot mugs
  and bumps. The model, trained on labeled press timing and synthetic table taps, is what separates them.
- **Real-time plumbing.** We fixed the reset button never registering, region crops crashing the
  optical flow, and shift-map images that never reached the dashboard (QNX caps UDP packets at 9 KB).

## What's next

- Natural backgrounds such as walls and brick instead of a dot screen.
- Methane and hydrogen. Methane rises where butane sinks, so the same physics applies but the
  signal is weaker. We couldn't bring natural gas indoors.
- A hardware watchdog behind the relay, so the valve drops out unless the supervisor keeps it alive.

## Repository

| Path | What's there |
|---|---|
| `mirage-starter/vision/` | `mirage_vision`: camera → BOS → model → verdicts, plus the Pi's web view |
| `mirage-starter/supervisor/` | The safety supervisor, and `mock_vision` for testing without a camera |
| `mirage-starter/tools/` | `mirage_ctl` (status and reset), `cpu_hog` (chaos demo), `run_mirage.sh` |
| `mirage-starter/mac/` | Dashboard, training, BOS prototype, dot screen (`dots.html`) |
| `mirage-starter/export/` | The trained model |
| `*.cpp` at the top level | QNX camera tools: `frame_stream` (browser stream) and `frame_grab` (saves a frame) |

The team's build notes, wiring and demo script are in [`mirage-starter/README.md`](mirage-starter/README.md).

## Running it

On the Pi (QNX 8, Raspberry Pi 5, Camera Module 3):

```sh
sudo apk add opencv-dev tflite-runtime tflite-runtime-dev flatbuffers-dev qnx-sf-base-dev
make mirage_vision CXX=clang++ EXTRA_CXXFLAGS=-I.
# Build the supervisor and tools with qcc (see mirage-starter/README.md), then:
sudo ./supervisor <dashboard IP>
on -p 20 ./mirage_vision -s <x,y,w,h of the dot screen> -i <dashboard IP> -m model.tflite -R
```

On the laptop:

```sh
pip install opencv-python numpy
python mirage-starter/mac/web_dashboard.py --host 0.0.0.0
```

Open `mirage-starter/mac/dots.html` full screen behind the area you're watching. Click **New
reference frame** with nothing in front of it, then **Open valve**.
