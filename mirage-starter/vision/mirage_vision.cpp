// mirage_vision.cpp: the Mirage vision process, built on Alex's camera code.   owner: Jaime
//
// Opens the Pi camera the way frame_stream does and runs every frame through mirage_on_frame(),
// which sends the supervisor a verdict 10 times a second. It needs OpenCV and TFLite, so it is
// built on the Pi (see the Makefile).
//
// Usage:
//   mirage_vision [-u unit] [-r height] [-w warmup] [-s x,y,w,h] [-m model] [-d dir] [-i mac_ip]
//                 [-p port] [-R]
//
//   -u  camera unit (default 4 = Pi Camera Module 3 in the Pi 5's second port; 3 = first port)
//   -r  use the smallest camera resolution at least this tall (default 864). Shifts are measured
//       in its pixels, so record and run with the same -r
//   -w  frames to skip before the reference frame, so auto-exposure settles (default 30)
//   -s  where the dot screen sits in the frame (default: the whole frame)
//   -m  TFLite model; it must load or mirage_vision won't start. Without -m, a big shift
//       counts as gas
//   -d  record fields to this folder, named after the label (mkdir -p it first)
//   -i  IP of the Mac running mac/dashboard.py, for shift-map thumbnails
//   -p  port for the browser view at http://<pi>:<port>/ (default 8080; 0 = off). It renders on
//       threads below the vision thread, and only while someone is watching
//   -R  rotate the browser view 180 degrees (camera mounted upside down); detection unaffected
//
// The first processed frame is saved to /tmp/frame.bmp with the dot screen boxed in red, so
// pull_frame.ps1 shows exactly what the engine sees. After a bump,
// `slay -s SIGUSR1 mirage_vision` takes a new reference frame and saves the frame again.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../../camera_frame.h"
#include "mirage_hook.hpp"
#include "web_view.hpp"

namespace {

using Clock = std::chrono::steady_clock;

const char kSnapshotPath[] = "/tmp/frame.bmp";      // where pull_frame.ps1 looks
const char kSnapshotTmp[] = "/tmp/frame.tmp.bmp";   // imwrite picks the format from the extension
const int kWebPriority = 10;                        // below vision (20) and cpu_hog's lighter run (15)

std::atomic<bool> gRunning{true};
std::atomic<bool> gNewReference{false};

struct Stats {
    unsigned frames = 0;
    unsigned verdicts = 0;
    unsigned gasVerdicts = 0;
    double worstMs = 0.0;     // slowest mirage_on_frame() call
    float peakPx = 0.f;       // largest peak shift in any verdict
};

struct VisionState {
    // Set before the viewfinder starts
    cv::Rect roi;
    const char* modelPath = nullptr;
    const char* dumpDir = nullptr;
    const char* dashboardIp = nullptr;

    // Camera thread only
    int warmup = 30;          // frames left to skip
    bool started = false;
    bool warnedUnsupported = false;
    Frame rgb;
    cv::Mat converted;
    WebViewFrame web;         // buffers for the browser view, swapped with its render thread

    // Shared with the main thread
    std::mutex mutex;
    Stats stats;
    cv::Mat snapshot;         // a processed frame waiting to be saved
    cv::Rect snapshotRoi;
};

// Gray view of a camera buffer, valid until the callback returns. NV12 (the Pi camera's
// default) starts with its Y plane, which is already gray, so the usual case copies nothing.
bool grayView(camera_buffer_t* buffer, VisionState* s, cv::Mat& gray)
{
    if (buffer->frametype == CAMERA_FRAMETYPE_NV12) {
        const auto& d = buffer->framedesc.nv12;
        gray = cv::Mat(static_cast<int>(d.height), static_cast<int>(d.width), CV_8UC1,
                       buffer->framebuf, d.stride);
        return true;
    }
    if (!convertToRgb(buffer, s->rgb)) {
        return false;
    }
    cv::Mat rgb(static_cast<int>(s->rgb.height), static_cast<int>(s->rgb.width), CV_8UC3,
                s->rgb.rgb.data());
    cv::cvtColor(rgb, s->converted, cv::COLOR_RGB2GRAY);
    gray = s->converted;
    return true;
}

// Runs on the camera thread with the first frame after warm-up, once the frame size is known
void start(VisionState* s, const cv::Mat& gray)
{
    if (s->roi.area() > 0) {
        cv::Rect inside = s->roi & cv::Rect(0, 0, gray.cols, gray.rows);
        if (inside.width < 16 || inside.height < 16) {
            std::printf("[vision] dot screen region is outside the %dx%d frame, using the whole frame\n",
                        gray.cols, gray.rows);
            inside = cv::Rect();
        } else if (inside != s->roi) {
            std::printf("[vision] dot screen region clipped to the frame: %d,%d,%d,%d\n",
                        inside.x, inside.y, inside.width, inside.height);
        }
        s->roi = inside;
    }

    // The chaos demo relies on vision outranking cpu_hog, so show what this thread really runs at
    sched_param sp{};
    int policy = 0;
    pthread_getschedparam(pthread_self(), &policy, &sp);
    if (s->roi.area() > 0) {
        std::printf("[vision] %dx%d frames, camera thread at priority %d, dot screen %d,%d,%d,%d\n",
                    gray.cols, gray.rows, sp.sched_priority, s->roi.x, s->roi.y, s->roi.width, s->roi.height);
    } else {
        std::printf("[vision] %dx%d frames, camera thread at priority %d, whole frame (no -s)\n",
                    gray.cols, gray.rows, sp.sched_priority);
    }
    std::fflush(stdout);
    mirage_init(s->roi, s->modelPath, s->dumpDir, s->dashboardIp);
    s->started = true;
}

// Runs on the camera's thread for every new buffer. The buffer is only valid during this call,
// so the whole pipeline runs here; anything that can wait is left to the main thread.
void onFrame(camera_handle_t handle, camera_buffer_t* buffer, void* arg)
{
    (void)handle;
    auto* s = static_cast<VisionState*>(arg);
    if (s->warmup > 0) {
        s->warmup--;
        return;
    }
    cv::Mat gray;
    if (!grayView(buffer, s, gray)) {
        if (!s->warnedUnsupported) {
            std::printf("[vision] unsupported frametype %d\n", (int)buffer->frametype);
            s->warnedUnsupported = true;
        }
        return;
    }
    bool wantSnapshot = false;
    if (!s->started) {
        start(s, gray);
        wantSnapshot = true;
    }
    if (gNewReference.exchange(false)) {
        mirage_reset_reference();
        wantSnapshot = true;
    }

    Clock::time_point t0 = Clock::now();
    mirage_verdict_t v{};
    bool produced = mirage_on_frame(gray, 0, &v);
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // A copy of the frame (well under a millisecond) is all the browser view costs this thread
    if (produced && web_view_wanted()) {
        WebViewFrame& w = s->web;
        gray.copyTo(w.gray);
        w.roi = s->roi;
        w.shiftMap = mirage_shift_map(256);
        w.verdict = v;
        w.valve = mirage_valve_state(&w.reason);
        w.procMs = ms;
        web_view_publish(w);
    }

    std::lock_guard<std::mutex> lock(s->mutex);
    if (wantSnapshot) {
        gray.copyTo(s->snapshot);   // written by the main thread: disk I/O here could delay verdicts
        s->snapshotRoi = s->roi;
    }
    Stats& st = s->stats;
    st.frames++;
    st.worstMs = std::max(st.worstMs, ms);
    if (produced) {
        st.verdicts++;
        if (v.cls == MIRAGE_GAS) {
            st.gasVerdicts++;
        }
        st.peakPx = std::max(st.peakPx, v.peak_px);
    }
}

// Boxes the dot screen in red and saves the frame. Writes to a temp file then renames, so a
// reader never sees a half-written image.
void saveSnapshot(const cv::Mat& gray, const cv::Rect& roi)
{
    cv::Mat img;
    cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);
    if (roi.area() > 0) {
        cv::rectangle(img, roi, cv::Scalar(0, 0, 255), 2);
    }
    if (!cv::imwrite(kSnapshotTmp, img) || std::rename(kSnapshotTmp, kSnapshotPath) != 0) {
        std::printf("[vision] could not save %s\n", kSnapshotPath);
        return;
    }
    std::printf("[vision] saved %s\n", kSnapshotPath);
}

long parseLong(const char* s, const char* name, long min, long max)
{
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < min || v > max) {
        std::printf("Invalid %s: '%s' (expected %ld-%ld)\n", name, s, min, max);
        std::exit(EXIT_FAILURE);
    }
    return v;
}

cv::Rect parseRect(const char* s)
{
    int x = 0, y = 0, w = 0, h = 0;
    char extra = 0;
    if (std::sscanf(s, "%d,%d,%d,%d%c", &x, &y, &w, &h, &extra) != 4 || x < 0 || y < 0 || w < 16 || h < 16) {
        std::printf("Invalid dot screen region: '%s' (expected pixels as x,y,width,height, e.g. -s 400,180,640,420;"
                    " width and height at least 16)\n", s);
        std::exit(EXIT_FAILURE);
    }
    return cv::Rect(x, y, w, h);
}

}  // namespace

int main(int argc, char* argv[])
{
    camera_unit_t unit = static_cast<camera_unit_t>(4);
    long height = 864;
    long port = 8080;
    bool rotate = false;
    VisionState state;

    int opt;
    while ((opt = getopt(argc, argv, "u:r:w:s:m:d:i:p:Rh")) != -1) {
        switch (opt) {
        case 'u': unit = static_cast<camera_unit_t>(parseLong(optarg, "unit", 1, CAMERA_UNIT_NUM_UNITS - 1)); break;
        case 'r': height = parseLong(optarg, "height", 0, 4096); break;
        case 'w': state.warmup = static_cast<int>(parseLong(optarg, "warmup", 0, 1000)); break;
        case 's': state.roi = parseRect(optarg); break;
        case 'm': state.modelPath = optarg; break;
        case 'd': state.dumpDir = optarg; break;
        case 'i': state.dashboardIp = optarg; break;
        case 'p': port = parseLong(optarg, "port", 0, 65535); break;
        case 'R': rotate = true; break;
        default:
            std::printf("Usage: %s [-u unit] [-r height] [-w warmup] [-s x,y,w,h] [-m model] [-d dir] [-i mac_ip]"
                        " [-p port] [-R]\n", argv[0]);
            return (opt == 'h') ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    // A model that was asked for must load, before the camera opens: running on without it
    // would quietly switch to the physics rule, which calls heat gas
    if (state.modelPath && !mirage_check_model(state.modelPath)) {
        std::printf("[vision] not starting: model %s did not load (reason above)\n", state.modelPath);
        return EXIT_FAILURE;
    }

    // Same signal handling as frame_stream: block the signals before any thread exists so every
    // thread (including libcamapi's) inherits the mask, and wait for them on one thread.
    // SIGUSR1 asks for a new reference frame.
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    sigaddset(&signals, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    std::thread([signals] {
        int sig = 0;
        while (sigwait(&signals, &sig) == 0 && sig == SIGUSR1) {
            gNewReference = true;
        }
        gRunning = false;
    }).detach();

    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    int err = camera_open(unit, CAMERA_MODE_RO | CAMERA_MODE_PWRITE, &handle);
    if (err != CAMERA_EOK || handle == CAMERA_HANDLE_INVALID) {
        std::printf("Failed to open CAMERA_UNIT_%d: err = %d (is the sensor service running?)\n",
                    (int)unit, err);
        return EXIT_FAILURE;
    }
    if (configureFrametype(handle) != CAMERA_EOK ||
        configureResolution(handle, static_cast<uint32_t>(height)) != CAMERA_EOK) {
        camera_close(handle);
        return EXIT_FAILURE;
    }
    err = camera_start_viewfinder(handle, onFrame, nullptr, &state);
    if (err != CAMERA_EOK) {
        std::printf("Failed to start CAMERA_UNIT_%d: err = %d\n", (int)unit, err);
        camera_close(handle);
        return EXIT_FAILURE;
    }
    std::printf("Mirage vision on CAMERA_UNIT_%d (Ctrl+C to stop, `slay -s SIGUSR1 mirage_vision` for a new reference)\n",
                (int)unit);
    // The page's "New reference frame" button does the same as SIGUSR1
    if (port > 0 && web_view_start(static_cast<int>(port), kWebPriority, rotate, [] { gNewReference = true; },
                                   mirage_request_reset)) {
        std::printf("[vision] browser view on http://qnxpi.local:%ld/\n", port);
    }
    std::fflush(stdout);

    // Saves snapshots and prints stats every 2 s, so the camera thread never waits on the disk
    // or the terminal
    Clock::time_point statsStart = Clock::now();
    while (gRunning.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        double elapsed = std::chrono::duration<double>(Clock::now() - statsStart).count();
        cv::Mat snapshot;
        cv::Rect roi;
        Stats st;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            std::swap(snapshot, state.snapshot);
            roi = state.snapshotRoi;
            if (elapsed >= 2.0) {
                st = state.stats;
                state.stats = Stats();
            }
        }
        if (!snapshot.empty()) {
            saveSnapshot(snapshot, roi);
        }
        if (elapsed < 2.0) {
            continue;
        }
        if (st.frames == 0) {
            std::printf("[vision] no frames from the camera (check the cable)\n");
        } else {
            std::printf("[vision] %4.1f fps  worst frame %5.1f ms  %u verdicts, %u gas  peak %.2f px\n",
                        st.frames / elapsed, st.worstMs, st.verdicts, st.gasVerdicts, st.peakPx);
        }
        std::fflush(stdout);
        statsStart = Clock::now();
    }

    std::printf("\nStopping...\n");
    std::fflush(stdout);
    camera_stop_viewfinder(handle);
    camera_close(handle);
    return EXIT_SUCCESS;
}
