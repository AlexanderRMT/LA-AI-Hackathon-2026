/**
 * @file frame_grab.cpp
 *
 * @brief Grabs frames from a QNX Sensor Framework camera and saves them as BMP
 *        images, so the camera can be checked over SSH without a monitor.
 *
 * Based on camera_example1_callback (Apache-2.0, BlackBerry Limited).
 *
 * Usage:
 *   frame_grab [-u unit] [-o path] [-r height] [-R] [-w warmup_frames] [-i interval_ms]
 *
 *   -u  camera unit (default 4 = Pi Camera Module 3 in the Pi 5's second port;
 *       3 = first port, 1 = test pattern)
 *   -o  output file (default /tmp/frame.bmp)
 *   -r  output height, aspect ratio kept (default 480: 854x480; 0 = full size)
 *   -R  rotate the picture 180 degrees (camera mounted upside down)
 *   -w  frames to skip so auto-exposure can settle (default 30)
 *   -i  keep running and overwrite the file every N ms (default 0 = one shot)
 */

#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>

#include "camera_frame.h"

namespace {

struct GrabState {
    std::mutex mutex;
    std::condition_variable cv;
    int framesToSkip = 0;
    uint32_t outHeight = kDefaultHeight;
    bool rotate180 = false;
    bool wantFrame = true;
    bool haveFrame = false;
    bool unsupported = false;
    camera_frametype_t badType = CAMERA_FRAMETYPE_UNSPECIFIED;
    Frame frame;
};

inline void putLe16(uint8_t* p, uint32_t v) { p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; }
inline void putLe32(uint8_t* p, uint32_t v) { putLe16(p, v); putLe16(p + 2, v >> 16); }

// Writes a 24-bit BMP (opens natively on Windows/macOS). Writes to a temp file
// then renames, so a reader never sees a half-written image.
bool writeBmp(const std::string& path, const Frame& frame)
{
    const uint32_t rowBytes = (frame.width * 3u + 3u) & ~3u;   // rows padded to 4 bytes
    const uint32_t dataSize = rowBytes * frame.height;
    uint8_t header[54] = {};
    header[0] = 'B';
    header[1] = 'M';
    putLe32(header + 2, 54u + dataSize);    // file size
    putLe32(header + 10, 54u);              // pixel data offset
    putLe32(header + 14, 40u);              // BITMAPINFOHEADER size
    putLe32(header + 18, frame.width);
    putLe32(header + 22, frame.height);     // positive = bottom-up rows
    putLe16(header + 26, 1u);               // planes
    putLe16(header + 28, 24u);              // bits per pixel
    putLe32(header + 34, dataSize);

    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        std::printf("Failed to open %s: errno %d\n", tmp.c_str(), errno);
        return false;
    }
    bool ok = std::fwrite(header, 1, sizeof(header), f) == sizeof(header);
    std::vector<uint8_t> row(rowBytes, 0);
    for (uint32_t y = frame.height; ok && y-- > 0;) {
        const uint8_t* src = &frame.rgb[static_cast<size_t>(y) * frame.width * 3u];
        for (uint32_t x = 0; x < frame.width; x++) {   // BMP stores B,G,R
            row[3 * x + 0] = src[3 * x + 2];
            row[3 * x + 1] = src[3 * x + 1];
            row[3 * x + 2] = src[3 * x + 0];
        }
        ok = std::fwrite(row.data(), 1, rowBytes, f) == rowBytes;
    }
    std::fclose(f);
    if (!ok) {
        std::printf("Short write to %s\n", tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::printf("Failed to rename %s -> %s: errno %d\n", tmp.c_str(), path.c_str(), errno);
        return false;
    }
    return true;
}

// Runs on the camera's thread for every new buffer
void onFrame(camera_handle_t handle, camera_buffer_t* buffer, void* arg)
{
    (void)handle;
    auto* state = static_cast<GrabState*>(arg);
    std::lock_guard<std::mutex> lock(state->mutex);

    if (state->framesToSkip > 0) {
        state->framesToSkip--;
        return;
    }
    if (!state->wantFrame) {
        return;
    }
    if (!convertToRgb(buffer, state->frame, state->outHeight, state->rotate180)) {
        state->unsupported = true;
        state->badType = buffer->frametype;
    }
    state->wantFrame = false;
    state->haveFrame = true;
    state->cv.notify_one();
}

long parseLong(const char* s, const char* name)
{
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0 || v == LONG_MAX) {
        std::printf("Invalid %s: '%s'\n", name, s);
        std::exit(EXIT_FAILURE);
    }
    return v;
}

}  // namespace

int main(int argc, char* argv[])
{
    camera_unit_t unit = static_cast<camera_unit_t>(4);
    std::string outPath = "/tmp/frame.bmp";
    long outHeight = kDefaultHeight;
    bool rotate180 = false;
    long warmup = 30;
    long intervalMs = 0;

    int opt;
    while ((opt = getopt(argc, argv, "u:o:r:Rw:i:h")) != -1) {
        switch (opt) {
        case 'u': unit = static_cast<camera_unit_t>(parseLong(optarg, "unit")); break;
        case 'o': outPath = optarg; break;
        case 'r': outHeight = parseLong(optarg, "height"); break;
        case 'R': rotate180 = true; break;
        case 'w': warmup = parseLong(optarg, "warmup"); break;
        case 'i': intervalMs = parseLong(optarg, "interval"); break;
        default:
            std::printf("Usage: %s [-u unit] [-o path] [-r height] [-R] [-w warmup_frames] [-i interval_ms]\n", argv[0]);
            return (opt == 'h') ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    if (unit <= CAMERA_UNIT_NONE || unit >= CAMERA_UNIT_NUM_UNITS) {
        std::printf("Camera unit out of range\n");
        return EXIT_FAILURE;
    }

    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    int err = camera_open(unit, CAMERA_MODE_RO | CAMERA_MODE_PWRITE, &handle);
    if (err != CAMERA_EOK || handle == CAMERA_HANDLE_INVALID) {
        std::printf("Failed to open CAMERA_UNIT_%d: err = %d (is the sensor service running?)\n",
                    (int)unit, err);
        return EXIT_FAILURE;
    }
    if (configureFrametype(handle) != CAMERA_EOK ||
        configureResolution(handle, static_cast<uint32_t>(outHeight)) != CAMERA_EOK) {
        camera_close(handle);
        return EXIT_FAILURE;
    }

    GrabState state;
    state.outHeight = static_cast<uint32_t>(outHeight);
    state.rotate180 = rotate180;
    state.framesToSkip = static_cast<int>(warmup);

    err = camera_start_viewfinder(handle, onFrame, nullptr, &state);
    if (err != CAMERA_EOK) {
        std::printf("Failed to start CAMERA_UNIT_%d: err = %d\n", (int)unit, err);
        camera_close(handle);
        return EXIT_FAILURE;
    }

    int exitCode = EXIT_SUCCESS;
    unsigned long count = 0;
    for (;;) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            if (!state.cv.wait_for(lock, std::chrono::seconds(10), [&] { return state.haveFrame; })) {
                std::printf("Timed out waiting for a frame (check the camera cable)\n");
                exitCode = EXIT_FAILURE;
                break;
            }
            if (state.unsupported) {
                std::printf("Unsupported frametype %d\n", (int)state.badType);
                exitCode = EXIT_FAILURE;
                break;
            }
            frame = std::move(state.frame);
            state.haveFrame = false;
        }

        if (!writeBmp(outPath, frame)) {
            exitCode = EXIT_FAILURE;
            break;
        }
        count++;
        std::printf("\rSaved %ux%u frame #%lu to %s", frame.width, frame.height, count, outPath.c_str());
        std::fflush(stdout);

        if (intervalMs == 0) {
            break;
        }
        usleep(static_cast<useconds_t>(intervalMs) * 1000u);
        std::lock_guard<std::mutex> lock(state.mutex);
        state.wantFrame = true;
    }
    std::printf("\n");

    camera_stop_viewfinder(handle);
    camera_close(handle);
    return exitCode;
}
