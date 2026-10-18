// web_view.cpp: a browser view of what the vision process sees.               owner: Alex
//
// The HTTP side follows frame_stream.cpp: one MJPEG stream shared by every viewer.
#include "web_view.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

#ifdef MSG_NOSIGNAL
const int kSendFlags = MSG_NOSIGNAL;
#else
const int kSendFlags = 0;
#endif

const int kViewWidth = 960;         // width of the whole-frame picture
const int kRegionWidth = 800;       // width of the region picture
const double kRegionMargin = 0.25;  // region picture: margin around the dot screen region, as a
                                    // fraction of its longer side
const int kBarHeight = 44;          // status bar under the picture

// The two pictures served: the whole camera frame (/stream), and the dot screen region with a
// margin of what's around it (/region). Each is rendered only while someone watches it.
enum View { kFull = 0, kRegion = 1, kViews = 2 };
const char* const kClassNames[] = {"clear", "gas", "heat", "motion"};
const char* const kReasonNames[] = {"", "boot", "leak", "deadline"};

const char kIndexHtml[] =
    "<!doctype html><html><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>Mirage Vision</title><style>"
    "html,body{margin:0;height:100%;background:#111;color:#ccc;font:14px system-ui,sans-serif}"
    "body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}"
    "img{max-width:100%;max-height:calc(100vh - 40px);object-fit:contain;background:#000}"
    "a{color:#8cf}button{font:inherit;padding:4px 12px;margin-left:8px;cursor:pointer}"
    "</style></head><body>"
    "<img src=\"/stream\" alt=\"Live Mirage vision view\">"
    "<div>Mirage vision on QNX &middot; region box turns red on gas &middot; "
    "<a href=\"/snapshot.jpg\" download=\"mirage.jpg\">save snapshot</a> &middot; "
    "<a href=\"/region\">region only</a>"
    "<button id=\"ref\" title=\"Clear the dot screen first: the next frame becomes the background\">"
    "New reference frame</button>"
    "<button id=\"open\" title=\"Asks the supervisor to reopen the valve, like mirage_ctl reset. "
    "It refuses while gas is in view.\">Open valve</button> <span id=\"msg\"></span></div>"
    "<script>"
    "function post(path,done){var m=document.getElementById('msg');m.textContent='...';"
    "fetch(path,{method:'POST'}).then(function(r){return r.text().then(function(t){"
    "m.textContent=done(r,t.trim());});}).catch(function(){m.textContent='Pi not reachable';});}"
    "document.getElementById('ref').onclick=function(){post('/reference',function(r){"
    "return r.ok?'new reference taken':'failed ('+r.status+')';});};"
    "document.getElementById('open').onclick=function(){post('/reset',function(r,t){"
    "return t==='accepted'?'valve open':t==='refused'?'refused: gas still in view':"
    "'supervisor not reachable';});};"
    "</script></body></html>";

struct State {
    int priority = 10;
    bool rotate = false;
    void (*onNewReference)() = nullptr;
    int (*onResetValve)() = nullptr;
    std::atomic<int> viewers[kViews]{};

    // Latest frame from the camera thread
    std::mutex frameMutex;
    std::condition_variable frameCv;
    WebViewFrame frame;
    bool fresh = false;

    // Latest JPEG of each view, shared by everyone watching it
    std::mutex jpegMutex;
    std::condition_variable jpegCv;
    std::shared_ptr<const std::vector<uchar>> jpeg[kViews];
    uint64_t seq[kViews] = {};
};

// Never destroyed: the detached render and HTTP threads may still be waiting on its mutexes and
// condition variables when the process exits, and destroying them under a waiter aborts the exit
State& g = *new State;

// Threads start at their creator's priority (the vision process runs at 20), so each one drops
// itself below the vision thread first. Threads it creates inherit the lower priority.
void lowerPriority()
{
    sched_param sp{};
    sp.sched_priority = g.priority;
    pthread_setschedparam(pthread_self(), SCHED_RR, &sp);
}

cv::Mat render(const WebViewFrame& f, View view)
{
    // The part of the camera frame to show: all of it, or the region with a margin around it
    const cv::Rect full(0, 0, f.gray.cols, f.gray.rows);
    cv::Rect src = full;
    if (view == kRegion && f.roi.area() > 0) {
        const int m = cvRound(kRegionMargin * std::max(f.roi.width, f.roi.height));
        src = cv::Rect(f.roi.x - m, f.roi.y - m, f.roi.width + 2 * m, f.roi.height + 2 * m) & full;
    }
    const int width = (view == kRegion) ? kRegionWidth : kViewWidth;
    const double scale = static_cast<double>(width) / src.width;
    cv::Mat small, img;
    cv::resize(f.gray(src), small, cv::Size(width, cvRound(src.height * scale)), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, img, cv::COLOR_GRAY2BGR);

    const cv::Rect frameRect(0, 0, img.cols, img.rows);
    cv::Rect roi = frameRect;
    if (f.roi.area() > 0) {
        roi = cv::Rect(cvRound((f.roi.x - src.x) * scale), cvRound((f.roi.y - src.y) * scale),
                       cvRound(f.roi.width * scale), cvRound(f.roi.height * scale)) & frameRect;
    }
    if (!f.shiftMap.empty() && roi.area() > 0) {
        cv::Mat map;
        cv::resize(f.shiftMap, map, roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat area = img(roi);
        cv::add(area, map, area);   // additive, so the dots stay visible and shifts glow on top
    }
    const bool gas = f.verdict.cls == MIRAGE_GAS;
    cv::rectangle(img, roi, gas ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0), 2);
    if (g.rotate) {
        cv::rotate(img, img, cv::ROTATE_180);
    }
    if (view == kRegion) {
        return img;   // the console around it already shows the valve and the verdict
    }

    cv::Mat out(img.rows + kBarHeight, img.cols, CV_8UC3, cv::Scalar(20, 20, 20));
    img.copyTo(out(frameRect));
    char valve[48];
    if (f.valve < 0) {
        std::snprintf(valve, sizeof valve, "NO SUPERVISOR");
    } else if (f.valve == 1) {
        std::snprintf(valve, sizeof valve, "VALVE OPEN");
    } else {
        std::snprintf(valve, sizeof valve, "VALVE CLOSED (%s)", f.reason < 4 ? kReasonNames[f.reason] : "?");
    }
    const cv::Scalar valveColor = f.valve == 1 ? cv::Scalar(90, 200, 120) : cv::Scalar(60, 70, 230);
    cv::putText(out, valve, cv::Point(12, img.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, valveColor, 2);
    char text[160];
    std::snprintf(text, sizeof text, "%s   p(gas) %.2f   peak %.2f px   %.0f ms/frame",
                  f.verdict.cls < 4 ? kClassNames[f.verdict.cls] : "?", f.verdict.p_gas,
                  f.verdict.peak_px, f.procMs);
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
    cv::putText(out, text, cv::Point(out.cols - textSize.width - 12, img.rows + 28), cv::FONT_HERSHEY_SIMPLEX,
                0.55, gas ? cv::Scalar(80, 80, 255) : cv::Scalar(225, 232, 238), 1);
    return out;
}

void renderLoop()
{
    lowerPriority();
    WebViewFrame f;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(g.frameMutex);
            g.frameCv.wait(lock, [] { return g.fresh; });
            std::swap(f, g.frame);
            g.fresh = false;
        }
        if (f.gray.empty()) {
            continue;
        }
        for (int v = 0; v < kViews; v++) {
            if (g.viewers[v].load() == 0) {
                continue;
            }
            auto jpeg = std::make_shared<std::vector<uchar>>();
            cv::imencode(".jpg", render(f, static_cast<View>(v)), *jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
            {
                std::lock_guard<std::mutex> lock(g.jpegMutex);
                g.jpeg[v] = std::move(jpeg);
                g.seq[v]++;
            }
            g.jpegCv.notify_all();
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

// Reads the request headers and returns the path of a GET or POST (POST paths are prefixed with
// "POST "), or "" if invalid. Request bodies are ignored.
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
    const bool post = std::strncmp(buf, "POST ", 5) == 0;
    if (!post && std::strncmp(buf, "GET ", 4) != 0) {
        return "";
    }
    const char* start = buf + (post ? 5 : 4);
    const char* end = std::strpbrk(start, " ?\r\n");
    if (end == nullptr) {
        return "";
    }
    return (post ? "POST " : "") + std::string(start, end);
}

// Waits for a JPEG of view v newer than lastSeq. Returns nullptr on timeout.
std::shared_ptr<const std::vector<uchar>> waitForJpeg(View v, uint64_t& lastSeq, std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(g.jpegMutex);
    if (!g.jpegCv.wait_for(lock, timeout, [&] { return g.seq[v] != lastSeq; })) {
        return nullptr;
    }
    lastSeq = g.seq[v];
    return g.jpeg[v];
}

uint64_t currentSeq(View v)
{
    std::lock_guard<std::mutex> lock(g.jpegMutex);
    return g.seq[v];
}

void streamMjpeg(int fd, View v)
{
    const char header[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (!sendAll(fd, header, sizeof(header) - 1)) {
        return;
    }
    g.viewers[v]++;
    uint64_t lastSeq = currentSeq(v);
    for (;;) {
        auto jpeg = waitForJpeg(v, lastSeq, std::chrono::milliseconds(1000));
        if (jpeg == nullptr) {
            continue;   // vision is busy or stalled (the chaos demo); keep the viewer connected
        }
        char part[128];
        int n = std::snprintf(part, sizeof(part),
                              "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                              jpeg->size());
        if (!sendAll(fd, part, static_cast<size_t>(n)) || !sendAll(fd, jpeg->data(), jpeg->size()) ||
            !sendAll(fd, "\r\n", 2)) {
            break;   // viewer disconnected
        }
    }
    g.viewers[v]--;
}

void sendSnapshot(int fd, View v)
{
    g.viewers[v]++;
    uint64_t lastSeq = currentSeq(v);
    auto jpeg = waitForJpeg(v, lastSeq, std::chrono::seconds(5));
    g.viewers[v]--;
    if (jpeg == nullptr) {
        const char msg[] = "No frame from the vision process\n";
        sendResponse(fd, "503 Service Unavailable", "text/plain", msg, sizeof(msg) - 1);
    } else {
        sendResponse(fd, "200 OK", "image/jpeg", jpeg->data(), jpeg->size());
    }
}

void serveClient(int fd)
{
    // Don't let a stalled viewer block its thread forever
    timeval timeout{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string path = readRequestPath(fd);
    if (path == "/" || path == "/index.html") {
        sendResponse(fd, "200 OK", "text/html; charset=utf-8", kIndexHtml, sizeof(kIndexHtml) - 1);
    } else if (path == "/stream") {
        streamMjpeg(fd, kFull);
    } else if (path == "/region") {
        streamMjpeg(fd, kRegion);
    } else if (path == "/snapshot.jpg") {
        sendSnapshot(fd, kFull);
    } else if (path == "/region.jpg") {
        sendSnapshot(fd, kRegion);
    } else if (path == "POST /reference") {
        if (g.onNewReference) {
            g.onNewReference();
        }
        std::printf("[web] new reference frame requested from the browser\n");
        std::fflush(stdout);
        const char msg[] = "ok\n";
        sendResponse(fd, "200 OK", "text/plain", msg, sizeof(msg) - 1);
    } else if (path == "POST /reset") {
        // The supervisor decides: it refuses while gas is in view, as for mirage_ctl reset
        const int r = g.onResetValve ? g.onResetValve() : -1;
        const char* body = r > 0 ? "accepted\n" : (r == 0 ? "refused\n" : "unreachable\n");
        std::printf("[web] valve reset requested from the browser: %s", body);
        std::fflush(stdout);
        sendResponse(fd, r > 0 ? "200 OK" : (r == 0 ? "409 Conflict" : "503 Service Unavailable"),
                     "text/plain", body, std::strlen(body));
    } else if (!path.empty()) {
        const char msg[] = "Not found\n";
        sendResponse(fd, "404 Not Found", "text/plain", msg, sizeof(msg) - 1);
    }
    close(fd);
}

void acceptLoop(int listenFd)
{
    lowerPriority();
    for (;;) {
        int fd = accept(listenFd, nullptr, nullptr);
        if (fd < 0) {
            if (errno != EINTR) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }
        std::thread(serveClient, fd).detach();
    }
}

}  // namespace

bool web_view_start(int port, int priority, bool rotate180, void (*onNewReference)(),
                    int (*onResetValve)())
{
    g.priority = priority;
    g.rotate = rotate180;
    g.onNewReference = onNewReference;
    g.onResetValve = onResetValve;
    std::signal(SIGPIPE, SIG_IGN);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::printf("[web] socket() failed: errno %d\n", errno);
        return false;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, 8) != 0) {
        std::printf("[web] port %d unavailable: errno %d (is frame_stream running?)\n", port, errno);
        close(fd);
        return false;
    }
    std::thread(renderLoop).detach();
    std::thread(acceptLoop, fd).detach();
    return true;
}

bool web_view_wanted()
{
    return g.viewers[kFull].load() + g.viewers[kRegion].load() > 0;
}

void web_view_publish(WebViewFrame& f)
{
    {
        std::lock_guard<std::mutex> lock(g.frameMutex);
        std::swap(g.frame, f);
        g.fresh = true;
    }
    g.frameCv.notify_one();
}
