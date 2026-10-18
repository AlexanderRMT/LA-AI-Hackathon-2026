// bos_engine.cpp                                                            owner: Jaime
#include "bos_engine.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

BosEngine::BosEngine(const BosParams& p) : p_(p)
{
    dis_ = cv::DISOpticalFlow::create(p_.dis_preset);
}

BosEngine::~BosEngine()
{
    stopDump();
}

bool BosEngine::loadModel(const std::string& path, int threads)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!model_) {
        std::fprintf(stderr, "[bos] could not load %s\n", path.c_str());
        return false;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interp_);
    if (!interp_) {
        std::fprintf(stderr, "[bos] interpreter build failed (op version too new for this TFLite?)\n");
        return false;
    }
    interp_->SetNumThreads(threads);
    if (interp_->AllocateTensors() != kTfLiteOk) {
        std::fprintf(stderr, "[bos] AllocateTensors failed\n");
        interp_.reset();
        return false;
    }
    const TfLiteTensor* in = interp_->input_tensor(0);
    if (in->type != kTfLiteFloat32 || in->dims->size != 4 || in->dims->data[1] != p_.grid ||
        in->dims->data[2] != p_.grid || in->dims->data[3] != 2 * p_.stack) {
        std::fprintf(stderr, "[bos] model input does not match BosParams (want float32 1x%dx%dx%d)\n",
                     p_.grid, p_.grid, 2 * p_.stack);
        interp_.reset();
        return false;
    }
    std::fprintf(stderr, "[bos] model loaded: %s\n", path.c_str());
    return true;
}

static float median_of(std::vector<float>& v)
{
    if (v.empty()) {
        return 0.f;
    }
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

bool BosEngine::process(const cv::Mat& gray_full, uint64_t frame_ns, BosResult& out)
{
    const uint64_t period_ns = static_cast<uint64_t>(1e9 / p_.proc_hz);
    if (!want_ref_ && last_proc_ns_ && frame_ns - last_proc_ns_ < period_ns) {
        return false;                                   // keep the processing rate fixed
    }
    last_proc_ns_ = frame_ns;

    const cv::Mat gray = p_.roi.area() > 0 ? gray_full(p_.roi) : gray_full;
    if (want_ref_ || ref_.empty() || ref_.size() != gray.size()) {
        gray.copyTo(ref_);
        roi_size_ = gray.size();
        want_ref_ = false;
        smooth_.release();
        history_.clear();
        return false;
    }

    dis_->calc(ref_, gray, flow_);                      // CV_32FC2, in pixels

    // 1. Remove whole-image shift (table bumps, camera sway): subtract the median flow.
    std::vector<float> fx, fy;
    fx.reserve(flow_.total() / 16 + 1);
    fy.reserve(flow_.total() / 16 + 1);
    for (int y = 0; y < flow_.rows; y += 4) {
        const cv::Vec2f* row = flow_.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow_.cols; x += 4) {
            fx.push_back(row[x][0]);
            fy.push_back(row[x][1]);
        }
    }
    const float mx = median_of(fx);
    const float my = median_of(fy);
    flow_ -= cv::Scalar(mx, my);
    out.bump_px = std::sqrt(mx * mx + my * my);

    // 2. Shrink to the model grid. Area averaging also cuts noise.
    cv::Mat small;
    cv::resize(flow_, small, cv::Size(p_.grid, p_.grid), 0, 0, cv::INTER_AREA);

    // 3. Temporal smoothing.
    if (smooth_.empty()) {
        small.copyTo(smooth_);
    } else {
        cv::addWeighted(small, p_.ema, smooth_, 1.0 - p_.ema, 0.0, smooth_);
    }

    std::vector<cv::Mat> ch;
    cv::split(smooth_, ch);
    cv::Mat mag;
    cv::magnitude(ch[0], ch[1], mag);
    double mn = 0, mxv = 0;
    cv::minMaxLoc(mag, &mn, &mxv);
    out.peak_px = static_cast<float>(mxv);
    out.frame_ns = frame_ns;

    history_.push_back(smooth_.clone());
    while (static_cast<int>(history_.size()) > p_.stack) {
        history_.pop_front();
    }

    if (dump_fields_) {
        fwrite(smooth_.ptr<float>(), sizeof(float), static_cast<size_t>(p_.grid) * p_.grid * 2, dump_fields_);
        std::fprintf(dump_meta_, "%llu,%llu,%.4f,%.4f\n", static_cast<unsigned long long>(dump_idx_++),
                     static_cast<unsigned long long>(frame_ns), out.peak_px, out.bump_px);
    }

    out.have_model_output = false;
    if (interp_ && static_cast<int>(history_.size()) == p_.stack) {
        float* in = interp_->typed_input_tensor<float>(0);
        const int G = p_.grid, C = 2 * p_.stack;
        for (int s = 0; s < p_.stack; s++) {
            const cv::Mat& f = history_[static_cast<size_t>(s)];
            for (int y = 0; y < G; y++) {
                const cv::Vec2f* row = f.ptr<cv::Vec2f>(y);
                for (int x = 0; x < G; x++) {
                    float* dst = in + (static_cast<size_t>(y) * G + x) * C + 2 * s;   // dx, dy of field s
                    dst[0] = std::max(-p_.clip_px, std::min(p_.clip_px, row[x][0])) * p_.scale;
                    dst[1] = std::max(-p_.clip_px, std::min(p_.clip_px, row[x][1])) * p_.scale;
                }
            }
        }
        if (interp_->Invoke() == kTfLiteOk) {
            const float* o = interp_->typed_output_tensor<float>(0);
            int best = 0;
            for (int i = 0; i < 4; i++) {
                out.probs[i] = o[i];
                if (o[i] > o[best]) {
                    best = i;
                }
            }
            out.cls = best;
            out.have_model_output = true;
        }
    }
    return true;
}

cv::Mat BosEngine::visual(int width) const
{
    const int height = roi_size_.width > 0 ? width * roi_size_.height / roi_size_.width : width * 2 / 3;
    if (smooth_.empty()) {
        return cv::Mat::zeros(height, width, CV_8UC3);
    }
    std::vector<cv::Mat> ch;
    cv::split(smooth_, ch);
    cv::Mat mag, m8, color, big;
    cv::magnitude(ch[0], ch[1], mag);
    mag.convertTo(m8, CV_8U, 255.0);                   // 1 px of shift = full brightness
    cv::applyColorMap(m8, color, cv::COLORMAP_INFERNO);
    cv::resize(color, big, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    return big;
}

bool BosEngine::startDump(const std::string& dir)
{
    stopDump();
    dump_fields_ = std::fopen((dir + "/fields.f32").c_str(), "wb");
    dump_meta_ = std::fopen((dir + "/meta.csv").c_str(), "w");
    if (!dump_fields_ || !dump_meta_) {
        std::fprintf(stderr, "[bos] cannot write to %s (mkdir -p it first)\n", dir.c_str());
        stopDump();
        return false;
    }
    std::fprintf(dump_meta_, "idx,frame_ns,peak_px,bump_px\n");
    dump_idx_ = 0;
    return true;
}

void BosEngine::stopDump()
{
    if (dump_fields_) {
        std::fclose(dump_fields_);
    }
    if (dump_meta_) {
        std::fclose(dump_meta_);
    }
    dump_fields_ = nullptr;
    dump_meta_ = nullptr;
}
