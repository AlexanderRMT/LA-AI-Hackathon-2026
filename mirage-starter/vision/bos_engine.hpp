// bos_engine.hpp: background-oriented schlieren plus TFLite verdicts.      owner: Jaime
//
// Feed it every camera frame. It keeps a reference frame of the dot screen, measures how
// far every dot appears to move (DIS optical flow), removes whole-image bumps, smooths,
// shrinks the field to the model grid, and runs the classifier on the last few fields.
// mac/bos_proto.py and mac/train.py copy these exact steps and constants.
#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

// TFLite model and interpreter. Defined in bos_engine.cpp: TFLite's class names are aliases,
// which can't be forward-declared here.
struct BosModel;

struct BosParams {
    cv::Rect roi;               // dot screen region in the camera frame; empty means the whole frame
    int   grid = 64;            // model input is grid x grid
    int   stack = 4;            // fields per model input, oldest first
    float ema = 0.5f;           // temporal smoothing of the field
    float clip_px = 4.0f;       // field values are clipped to +-clip_px ...
    float scale = 0.5f;         // ... then multiplied by scale (must match mac/train.py)
    float proc_hz = 10.0f;      // processing rate; training data must use the same rate
    int   dis_preset = cv::DISOpticalFlow::PRESET_MEDIUM;
};

struct BosResult {
    uint64_t frame_ns = 0;
    float peak_px = 0.f;        // largest dot shift after bump removal and smoothing
    float bump_px = 0.f;        // whole-image shift that was removed
    float probs[4] = {1.f, 0.f, 0.f, 0.f};   // clear, gas, heat, motion
    int   cls = 0;
    bool  have_model_output = false;
};

class BosEngine {
public:
    explicit BosEngine(const BosParams& p);
    ~BosEngine();

    bool loadModel(const std::string& tflite_path, int threads = 2);
    bool hasModel() const { return static_cast<bool>(model_); }
    void resetReference() { want_ref_ = true; }

    // gray: 8-bit single-channel full frame. Returns true when it produced a new result.
    bool process(const cv::Mat& gray, uint64_t frame_ns, BosResult& out);

    // Colored picture of the latest shift field for the GTK window or the dashboard.
    cv::Mat visual(int width = 480) const;

    // Appends every processed field to <dir>/fields.f32 and a row to <dir>/meta.csv.
    // Name the folder after the label, for example rec/gas_01 or rec/heat_iron_02.
    bool startDump(const std::string& dir);
    void stopDump();

private:
    BosParams p_;
    cv::Ptr<cv::DISOpticalFlow> dis_;
    cv::Mat ref_, cur_, flow_, smooth_;     // ref, cur: CV_8U, flows: CV_32FC2
    std::deque<cv::Mat> history_;           // grid x grid CV_32FC2, oldest first
    cv::Size roi_size_;
    bool want_ref_ = true;
    uint64_t next_proc_ns_ = 0;
    std::unique_ptr<BosModel> model_;       // null until a model loads and matches BosParams
    FILE* dump_fields_ = nullptr;
    FILE* dump_meta_ = nullptr;
    uint64_t dump_idx_ = 0;
};
