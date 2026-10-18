// web_view.hpp: a browser view of what the vision process sees.               owner: Alex
//
// Serves http://<pi>:<port>/ with the camera frame, the shift map glowing over the dot screen
// region, and the latest verdict and valve state; /region streams just the region with a margin
// around it (the Mac console shows that one). Rendering, JPEG encoding and the HTTP clients
// all run on threads below the vision thread, so the view can stall but never delays a verdict.
#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

extern "C" {
#include "../common/mirage_msgs.h"
}

struct WebViewFrame {
    cv::Mat gray;                   // camera frame
    cv::Rect roi;                   // dot screen region in gray; empty means the whole frame
    cv::Mat shiftMap;               // BGR shift map of the region, any size
    mirage_verdict_t verdict{};
    int valve = -1;                 // supervisor's last reply: 1 open, 0 closed, -1 not reachable
    uint32_t reason = 0;            // enum mirage_trip, while closed
    double procMs = 0.0;            // how long mirage_on_frame() took
};

// Starts the server and render threads at `priority`. rotate180 turns the picture only, for a
// camera mounted upside down; detection and region coordinates are unaffected. The page's
// "New reference frame" button calls onNewReference from an HTTP thread, so it should only set
// a flag for the camera thread. Its "Open valve" button (POST /reset) calls onResetValve, which
// asks the supervisor and returns 1 accepted, 0 refused, -1 not reachable.
bool web_view_start(int port, int priority, bool rotate180, void (*onNewReference)(),
                    int (*onResetValve)());

// True while a browser is watching. Skip building frames otherwise.
bool web_view_wanted();

// Hands a frame to the render thread. Swaps buffers with f, so it costs no copy and f's old
// buffers come back for reuse.
void web_view_publish(WebViewFrame& f);
