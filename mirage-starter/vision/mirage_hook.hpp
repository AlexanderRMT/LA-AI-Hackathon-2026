// mirage_hook.hpp: the vision pipeline as the camera callback sees it.       owner: Jaime
#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

extern "C" {
#include "../common/mirage_msgs.h"
}

// Call once before the first frame. screen_roi: where the dot screen sits in the frame (empty
// means the whole frame). model_path / dump_dir / dashboard_ip may be null. If model_path is set
// but doesn't load, no verdicts are produced at all (the supervisor's deadline closes the valve)
// rather than falling back to the physics rule, which calls heat gas.
void mirage_init(const cv::Rect& screen_roi, const char* model_path, const char* dump_dir,
                 const char* dashboard_ip);

// True if model_path loads and fits the engine (readable, ops supported, float32 1x64x64x8 in).
// The reason it doesn't is printed. mirage_vision checks this before opening the camera.
bool mirage_check_model(const char* model_path);

// Takes a new reference frame on the next mirage_on_frame(). Call it from the camera thread.
void mirage_reset_reference();

// frame: gray, BGR, or BGRA. frame_ns: CLOCK_MONOTONIC capture time, or 0 for now.
// Returns true when the frame produced a verdict (10 per second) and copies it to *verdict.
bool mirage_on_frame(const cv::Mat& frame, uint64_t frame_ns, mirage_verdict_t* verdict = nullptr);

// The supervisor's reply to the last verdict: 1 valve open, 0 closed (reason in *reason),
// -1 not reachable.
int mirage_valve_state(uint32_t* reason = nullptr);

// Colour shift map of the dot screen region, width pixels wide. Call it from the camera thread.
cv::Mat mirage_shift_map(int width);

// Asks the supervisor to reopen the valve, exactly like `mirage_ctl reset`; its rules decide.
// Safe from any thread (uses its own connection). 1 accepted, 0 refused, -1 not reachable.
int mirage_request_reset();
