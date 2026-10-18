/**
 * @file frame_stream.cpp
 *
 * @brief Streams a QNX Sensor Framework camera as MJPEG over HTTP, so the feed
 *        can be watched live in a laptop browser.
 *
 * Usage:
 *   frame_stream [-u unit] [-p port] [-r height] [-q quality] [-f max_fps] [-j threads]
 *
 *   -u  camera unit (default 4 = Pi Camera Module 3 in the Pi 5's second port;
 *       3 = first port, 1 = test pattern)
 *   -p  HTTP port (default 8080)
 *   -r  output height, aspect ratio kept (default 480: 854x480; 0 = full size)
 *   -q  JPEG quality 1-100 (default 75)
 *   -f  max frames per second to encode (default 30, the camera's rate)
 *   -j  JPEG encoder threads (default 3: as fast as 4 on a Pi 5, leaves a core free)
 *
 * Then open http://qnxpi.local:8080/ in a browser. Frames are only converted
 * and encoded while at least one viewer is connected.
 *
 *   /              viewer page
 *   /stream        multipart MJPEG stream
 *   /snapshot.jpg  single frame
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include "camera_frame.h"
#include "jpeg_encoder.h"

namespace {

using Clock = std::chrono::steady_clock;

#ifdef MSG_NOSIGNAL
const int kSendFlags = MSG_NOSIGNAL;
#else
const int kSendFlags = 0;
#endif

const char kIndexHtml[] =
    "<!doctype html><html><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>QNX Camera</title><style>"
    "html,body{margin:0;height:100%;background:#111;color:#ccc;font:14px system-ui,sans-serif}"
    "body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}"
    "img{max-width:100%;max-height:calc(100vh - 40px);object-fit:contain;background:#000}"
    "a{color:#8cf}</style></head><body>"
    "<img src=\"/stream\" alt=\"Live camera stream\">"
    "<div>Live from QNX &middot; <a href=\"/snapshot.jpg\" download=\"snapshot.jpg\">save snapshot</a></div>"
    "</body></html>";

std::atomic<bool> gRunning{true};

// Latest converted frame, handed from the camera callback to the encoder thread
struct RawSlot {
    std::mutex mutex;
    std::condition_variable cv;
    Yuv420Image frame;
    bool fresh = false;
};

// Latest encoded JPEG, shared by every HTTP client
struct JpegSlot {
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<const std::vector<uint8_t>> jpeg;
    uint64_t seq = 0;
};

struct StreamState {
    uint32_t outHeight = kDefaultHeight;
    Clock::duration minInterval{};
    std::atomic<int> viewers{0};
    std::atomic<bool> warnedUnsupported{false};

    // Camera thread only
    Clock::time_point nextDue{};
    Yuv420Image spare;

    RawSlot raw;
    JpegSlot jpeg;
};

// Runs on the camera's thread for every new buffer. The buffer is only valid
// during this call, so it is converted (and downscaled) here.
void onFrame(camera_handle_t handle, camera_buffer_t* buffer, void* arg)
{
    (void)handle;
    auto* s = static_cast<StreamState*>(arg);
    if (s->viewers.load() == 0) {
        return;
    }
    // Advance a deadline instead of comparing to the last frame's time, so
    // camera jitter doesn't skip an extra frame (30 fps / 15 cap would give 10)
    Clock::time_point now = Clock::now();
    if (now < s->nextDue) {
        return;
    }
    s->nextDue += s->minInterval;
    if (s->nextDue < now) {
        s->nextDue = now;   // fell behind (e.g. first viewer after idle); don't burst
    }

    if (!convertToYuv420(buffer, s->spare, s->outHeight)) {
        if (!s->warnedUnsupported.exchange(true)) {
            std::printf("\nUnsupported frametype %d\n", (int)buffer->frametype);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(s->raw.mutex);
        std::swap(s->spare, s->raw.frame);   // spare keeps the old buffer for reuse
        s->raw.fresh = true;
    }
    s->raw.cv.notify_one();
}

// Encodes the newest frame to JPEG and publishes it; prints stats every 2s
void encodeLoop(StreamState* s, int quality, unsigned threads)
{
    JpegEncoder encoder(quality);
    Yuv420Image frame;
    unsigned frames = 0;
    double encodeMs = 0.0;
    size_t bytes = 0;
    Clock::time_point statsStart = Clock::now();

    while (gRunning.load()) {
        {
            std::unique_lock<std::mutex> lock(s->raw.mutex);
            s->raw.cv.wait_for(lock, std::chrono::milliseconds(200),
                               [&] { return s->raw.fresh || !gRunning.load(); });
            if (!s->raw.fresh) {
                continue;
            }
            std::swap(frame, s->raw.frame);
            s->raw.fresh = false;
        }

        Clock::time_point t0 = Clock::now();
        auto jpeg = std::make_shared<std::vector<uint8_t>>();
        encoder.encode(frame, *jpeg, threads);
        encodeMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        bytes += jpeg->size();
        frames++;
        {
            std::lock_guard<std::mutex> lock(s->jpeg.mutex);
            s->jpeg.jpeg = std::move(jpeg);
            s->jpeg.seq++;
        }
        s->jpeg.cv.notify_all();

        double elapsed = std::chrono::duration<double>(Clock::now() - statsStart).count();
        if (elapsed >= 2.0) {
            std::printf("\r%ux%u  %4.1f fps  encode %5.1f ms  %4zu KB/frame  viewers %d   ",
                        frame.width, frame.height, frames / elapsed, encodeMs / frames,
                        bytes / frames / 1024u, s->viewers.load());
            std::fflush(stdout);
            frames = 0;
            encodeMs = 0.0;
            bytes = 0;
            statsStart = Clock::now();
        }
    }
}

bool sendAll(int fd, const void* data, size_t len)
{
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = send(fd, p, len, kSendFlags);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool sendAll(int fd, const std::string& text)
{
    return sendAll(fd, text.data(), text.size());
}

void sendResponse(int fd, const char* status, const char* contentType, const void* body, size_t len)
{
    char header[256];
    int n = std::snprintf(header, sizeof(header),
                          "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                          "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                          status, contentType, len);
    if (sendAll(fd, header, static_cast<size_t>(n))) {
        sendAll(fd, body, len);
    }
}

// Reads the request headers and returns the path of a GET, or "" if invalid
std::string readRequestPath(int fd)
{
    char buf[2048];
    size_t used = 0;
    while (used < sizeof(buf) - 1) {
        ssize_t n = recv(fd, buf + used, sizeof(buf) - 1 - used, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return "";
        }
        used += static_cast<size_t>(n);
        buf[used] = '\0';
        if (std::strstr(buf, "\r\n\r\n") != nullptr) {
            break;
        }
    }
    buf[used] = '\0';
    if (std::strncmp(buf, "GET ", 4) != 0) {
        return "";
    }
    const char* start = buf + 4;
    const char* end = std::strpbrk(start, " ?\r\n");
    return (end != nullptr) ? std::string(start, end) : "";
}

// Waits for a JPEG newer than lastSeq. Returns nullptr on timeout or shutdown.
std::shared_ptr<const std::vector<uint8_t>> waitForJpeg(StreamState* s, uint64_t& lastSeq,
                                                         std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(s->jpeg.mutex);
    bool ready = s->jpeg.cv.wait_for(lock, timeout, [&] {
        return s->jpeg.seq != lastSeq || !gRunning.load();
    });
    if (!ready || !gRunning.load()) {
        return nullptr;
    }
    lastSeq = s->jpeg.seq;
    return s->jpeg.jpeg;
}

uint64_t currentSeq(StreamState* s)
{
    std::lock_guard<std::mutex> lock(s->jpeg.mutex);
    return s->jpeg.seq;
}

void streamMjpeg(int fd, StreamState* s)
{
    const std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (!sendAll(fd, header)) {
        return;
    }

    s->viewers++;
    uint64_t lastSeq = currentSeq(s);   // skip any stale frame from a previous viewer
    while (gRunning.load()) {
        auto jpeg = waitForJpeg(s, lastSeq, std::chrono::milliseconds(500));
        if (jpeg == nullptr) {
            continue;   // camera hasn't delivered yet; re-check gRunning
        }
        char part[128];
        int n = std::snprintf(part, sizeof(part),
                              "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                              jpeg->size());
        if (!sendAll(fd, part, static_cast<size_t>(n)) ||
            !sendAll(fd, jpeg->data(), jpeg->size()) ||
            !sendAll(fd, "\r\n", 2)) {
            break;   // viewer disconnected
        }
    }
    s->viewers--;
}

void sendSnapshot(int fd, StreamState* s)
{
    s->viewers++;
    uint64_t lastSeq = currentSeq(s);
    auto jpeg = waitForJpeg(s, lastSeq, std::chrono::seconds(5));
    s->viewers--;

    if (jpeg == nullptr) {
        const char msg[] = "No frame from camera\n";
        sendResponse(fd, "503 Service Unavailable", "text/plain", msg, sizeof(msg) - 1);
    } else {
        sendResponse(fd, "200 OK", "image/jpeg", jpeg->data(), jpeg->size());
    }
}

void serveClient(int fd, StreamState* s)
{
    // Don't let a stalled viewer block its thread forever
    timeval timeout{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string path = readRequestPath(fd);
    if (path == "/" || path == "/index.html") {
        sendResponse(fd, "200 OK", "text/html; charset=utf-8", kIndexHtml, sizeof(kIndexHtml) - 1);
    } else if (path == "/stream") {
        streamMjpeg(fd, s);
    } else if (path == "/snapshot.jpg") {
        sendSnapshot(fd, s);
    } else if (!path.empty()) {
        const char msg[] = "Not found\n";
        sendResponse(fd, "404 Not Found", "text/plain", msg, sizeof(msg) - 1);
    }
    close(fd);
}

int openListenSocket(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::printf("socket() failed: errno %d\n", errno);
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::printf("bind() to port %d failed: errno %d (already in use?)\n", port, errno);
        close(fd);
        return -1;
    }
    if (listen(fd, 8) != 0) {
        std::printf("listen() failed: errno %d\n", errno);
        close(fd);
        return -1;
    }
    return fd;
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

}  // namespace

int main(int argc, char* argv[])
{
    camera_unit_t unit = static_cast<camera_unit_t>(4);
    int port = 8080;
    long outHeight = kDefaultHeight;
    long quality = 75;
    long fps = 30;
    long threads = 3;

    int opt;
    while ((opt = getopt(argc, argv, "u:p:r:q:f:j:h")) != -1) {
        switch (opt) {
        case 'u': unit = static_cast<camera_unit_t>(parseLong(optarg, "unit", 1, CAMERA_UNIT_NUM_UNITS - 1)); break;
        case 'p': port = static_cast<int>(parseLong(optarg, "port", 1, 65535)); break;
        case 'r': outHeight = parseLong(optarg, "height", 0, 4096); break;
        case 'q': quality = parseLong(optarg, "quality", 1, 100); break;
        case 'f': fps = parseLong(optarg, "fps", 1, 120); break;
        case 'j': threads = parseLong(optarg, "threads", 1, 16); break;
        default:
            std::printf("Usage: %s [-u unit] [-p port] [-r height] [-q quality] [-f max_fps] [-j threads]\n", argv[0]);
            return (opt == 'h') ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    // Handle Ctrl+C on a dedicated thread: block the signals here so every
    // thread created later (including libcamapi's) inherits the mask.
    // Shells start background jobs with SIGINT ignored, and ignored signals
    // never reach sigwait, so restore the defaults first.
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    sigset_t stopSignals;
    sigemptyset(&stopSignals);
    sigaddset(&stopSignals, SIGINT);
    sigaddset(&stopSignals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stopSignals, nullptr);
    std::signal(SIGPIPE, SIG_IGN);
    std::thread([stopSignals] {
        int sig = 0;
        sigwait(&stopSignals, &sig);
        gRunning = false;
    }).detach();

    int listenFd = openListenSocket(port);
    if (listenFd < 0) {
        return EXIT_FAILURE;
    }

    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    int err = camera_open(unit, CAMERA_MODE_RO | CAMERA_MODE_PWRITE, &handle);
    if (err != CAMERA_EOK || handle == CAMERA_HANDLE_INVALID) {
        std::printf("Failed to open CAMERA_UNIT_%d: err = %d (is the sensor service running?)\n",
                    (int)unit, err);
        close(listenFd);
        return EXIT_FAILURE;
    }
    if (configureFrametype(handle) != CAMERA_EOK ||
        configureResolution(handle, static_cast<uint32_t>(outHeight)) != CAMERA_EOK) {
        camera_close(handle);
        close(listenFd);
        return EXIT_FAILURE;
    }

    static StreamState state;   // outlives detached client threads during shutdown
    state.outHeight = static_cast<uint32_t>(outHeight);
    state.minInterval = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));

    std::thread encoder(encodeLoop, &state, static_cast<int>(quality), static_cast<unsigned>(threads));

    err = camera_start_viewfinder(handle, onFrame, nullptr, &state);
    if (err != CAMERA_EOK) {
        std::printf("Failed to start CAMERA_UNIT_%d: err = %d\n", (int)unit, err);
        gRunning = false;
        encoder.join();
        camera_close(handle);
        close(listenFd);
        return EXIT_FAILURE;
    }

    std::printf("Streaming CAMERA_UNIT_%d on http://qnxpi.local:%d/  (%ld fps max, %ld encoder threads; Ctrl+C to stop)\n",
                (int)unit, port, fps, threads);
    std::fflush(stdout);

    // Poll so Ctrl+C is noticed even while no one is connecting
    while (gRunning.load()) {
        pollfd pfd{listenFd, POLLIN, 0};
        int ready = poll(&pfd, 1, 500);
        if (ready <= 0) {
            continue;
        }
        int client = accept(listenFd, nullptr, nullptr);
        if (client < 0) {
            continue;
        }
        std::thread(serveClient, client, &state).detach();
    }

    std::printf("\nStopping...\n");
    std::fflush(stdout);
    close(listenFd);
    state.raw.cv.notify_all();
    state.jpeg.cv.notify_all();
    encoder.join();
    camera_stop_viewfinder(handle);
    camera_close(handle);
    return EXIT_SUCCESS;
}
