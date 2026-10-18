// mirage_hook.cpp: glue between the camera callback and the BOS engine.     owner: Jaime
//
// mirage_vision.cpp calls mirage_init() once, then mirage_on_frame() for every camera frame.
// Until Gio's model arrives it uses a physics-only rule (a big shift counts as gas), so the
// whole loop to the supervisor can be tested from the first hour.
#include "mirage_hook.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bos_engine.hpp"
#include "verdict_client.hpp"

static BosEngine* g_engine = nullptr;
static VerdictClient g_client;
static int g_thumb_sock = -1;
static sockaddr_in g_thumb_dst;
static bool g_thumb_warned = false;             // report a failed thumbnail send once
static const size_t kMaxThumbBytes = 9216;      // QNX default net.inet.udp.maxdgram
static const float kPhysicsGasPx = 0.30f;      // fallback rule before the model exists
static int g_link = -1;                         // supervisor reachable? -1 until the first verdict
static mirage_status_t g_status;                // supervisor's reply to the last verdict
static bool g_model_missing = false;            // -m was given but didn't load: send no verdicts

static uint64_t mono_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

void mirage_init(const cv::Rect& screen_roi, const char* model_path, const char* dump_dir,
                 const char* dashboard_ip)
{
    BosParams p;
    p.roi = screen_roi;
    g_engine = new BosEngine(p);
    if (model_path && !g_engine->loadModel(model_path)) {
        // Never fall back to the physics rule when a model was asked for: it calls heat gas.
        std::fprintf(stderr, "[vision] model %s did not load: sending no verdicts, so the valve "
                             "closes on the supervisor's deadline\n", model_path);
        g_model_missing = true;
    }
    if (dump_dir) {
        g_engine->startDump(dump_dir);
    }
    if (dashboard_ip) {                         // JPEG thumbnails of the shift map on UDP 5006
        g_thumb_sock = socket(AF_INET, SOCK_DGRAM, 0);
        std::memset(&g_thumb_dst, 0, sizeof g_thumb_dst);
        g_thumb_dst.sin_family = AF_INET;
        g_thumb_dst.sin_port = htons(5006);
        if (g_thumb_sock == -1 || inet_pton(AF_INET, dashboard_ip, &g_thumb_dst.sin_addr) != 1) {
            g_thumb_sock = -1;
        }
    }
}

bool mirage_check_model(const char* model_path)
{
    BosParams p;
    BosEngine probe(p);
    return probe.loadModel(model_path);
}

void mirage_reset_reference()
{
    if (g_engine) {
        g_engine->resetReference();
    }
}

bool mirage_on_frame(const cv::Mat& frame, uint64_t frame_ns, mirage_verdict_t* verdict)
{
    if (!g_engine || g_model_missing) {
        return false;
    }
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame;                           // NV12's Y plane is already gray: zero-cost
    }
    BosResult r;
    if (!g_engine->process(gray, frame_ns ? frame_ns : mono_ns(), r)) {
        return false;
    }
    mirage_verdict_t v;
    std::memset(&v, 0, sizeof v);
    v.frame_ns = r.frame_ns;
    v.peak_px = r.peak_px;
    if (r.have_model_output) {
        v.cls = static_cast<uint8_t>(r.cls);
        v.p_gas = r.probs[MIRAGE_GAS];
        v.conf = r.probs[r.cls];
    } else {
        const bool gas = r.peak_px > kPhysicsGasPx;
        v.cls = gas ? MIRAGE_GAS : MIRAGE_CLEAR;
        v.p_gas = gas ? 0.9f : 0.05f;
        v.conf = 0.5f;
    }
    v.sent_ns = mono_ns();
    mirage_status_t st;
    const int link = g_client.send(v, &st) ? 1 : 0;
    if (link) {
        g_status = st;
    }
    if (link != g_link) {                       // report changes, not every verdict
        std::fprintf(stderr, link ? "[vision] connected to the supervisor\n"
                                  : "[vision] supervisor not reachable, verdicts are dropped\n");
        g_link = link;
    }

    if (g_thumb_sock != -1) {
        // QNX refuses UDP datagrams over net.inet.udp.maxdgram (9216 bytes by default), so step
        // the quality down until the thumbnail fits rather than sending nothing
        const cv::Mat img = g_engine->visual(360);
        std::vector<uchar> jpg;
        for (int quality : {70, 50, 35, 20}) {
            cv::imencode(".jpg", img, jpg, {cv::IMWRITE_JPEG_QUALITY, quality});
            if (jpg.size() <= kMaxThumbBytes) {
                break;
            }
        }
        const bool sent = jpg.size() <= kMaxThumbBytes &&
                          sendto(g_thumb_sock, jpg.data(), jpg.size(), 0,
                                 reinterpret_cast<sockaddr*>(&g_thumb_dst), sizeof g_thumb_dst) != -1;
        if (!sent && !g_thumb_warned) {
            std::fprintf(stderr, "[vision] shift-map thumbnail not sent (%zu bytes): %s\n", jpg.size(),
                         jpg.size() > kMaxThumbBytes ? "too big for one UDP datagram" : std::strerror(errno));
            g_thumb_warned = true;
        }
    }
    if (verdict) {
        *verdict = v;
    }
    return true;
}

int mirage_valve_state(uint32_t* reason)
{
    if (g_link != 1) {
        return -1;
    }
    if (reason) {
        *reason = g_status.reason;
    }
    return g_status.valve_open ? 1 : 0;
}

cv::Mat mirage_shift_map(int width)
{
    return g_engine ? g_engine->visual(width) : cv::Mat();
}

int mirage_request_reset()
{
    // Its own connection: g_client belongs to the camera thread
    const int coid = name_open(MIRAGE_SUP_NAME, 0);
    if (coid == -1) {
        return -1;
    }
    mirage_cmd_t cmd;
    std::memset(&cmd, 0, sizeof cmd);
    cmd.type = MIRAGE_MSG_RESET;
    mirage_status_t st;
    std::memset(&st, 0, sizeof st);
    const int rc = MsgSend(coid, &cmd, sizeof cmd, &st, sizeof st);
    name_close(coid);
    if (rc == -1) {
        return -1;
    }
    return st.reset_ok ? 1 : 0;
}
