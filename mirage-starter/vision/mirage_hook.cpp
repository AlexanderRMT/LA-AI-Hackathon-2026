// mirage_hook.cpp: drop-in glue for the QNX AI workshop's camera callback.   owner: Jaime
//
// The workshop already turns Sensor Framework frames into cv::Mat. Call mirage_init() once,
// then mirage_on_frame() from that callback in place of the palm and hand-landmark models.
// Until Gio's model arrives it uses a physics-only rule (a big shift counts as gas), so the
// whole loop to the supervisor can be tested from the first hour.
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
static const float kPhysicsGasPx = 0.30f;      // fallback rule before the model exists

static uint64_t mono_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// screen_roi: where the dot screen sits in the frame. model_path / dump_dir / dashboard_ip may be null.
void mirage_init(const cv::Rect& screen_roi, const char* model_path, const char* dump_dir,
                 const char* dashboard_ip)
{
    BosParams p;
    p.roi = screen_roi;
    g_engine = new BosEngine(p);
    if (model_path) {
        g_engine->loadModel(model_path);
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

void mirage_reset_reference()
{
    if (g_engine) {
        g_engine->resetReference();
    }
}

// frame: whatever the workshop callback has (gray, BGR, or BGRA). frame_ns: its timestamp, or 0.
void mirage_on_frame(const cv::Mat& frame, uint64_t frame_ns)
{
    if (!g_engine) {
        return;
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
        return;
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
    g_client.send(v, &st);

    if (g_thumb_sock != -1) {
        std::vector<uchar> jpg;
        cv::imencode(".jpg", g_engine->visual(360), jpg, {cv::IMWRITE_JPEG_QUALITY, 70});
        if (jpg.size() < 60000) {
            sendto(g_thumb_sock, jpg.data(), jpg.size(), 0,
                   reinterpret_cast<sockaddr*>(&g_thumb_dst), sizeof g_thumb_dst);
        }
    }
}
