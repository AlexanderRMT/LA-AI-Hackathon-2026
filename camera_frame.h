/**
 * @file camera_frame.h
 *
 * @brief Helpers shared by the camera tools: frametype setup and conversion of
 *        Sensor Framework buffers to packed RGB24.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <camera/camera_api.h>

#include "jpeg_encoder.h"

struct Frame {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;   // packed RGB24, width * height * 3
};

// Default output height for the camera tools (480p)
constexpr uint32_t kDefaultHeight = 480;

/**
 * @brief Converts a camera buffer to packed RGB24, downscaled (nearest
 *        neighbour) to @p outHeight rows with the aspect ratio kept and the
 *        width rounded to even, e.g. 1536x864 -> 854x480. 0, or a height at or
 *        above the source's, keeps the full size.
 *
 * Only valid inside the viewfinder callback, while @p buffer is still owned by
 * the caller. Returns false for frametypes it can't convert.
 */
bool convertToRgb(const camera_buffer_t* buffer, Frame& out, uint32_t outHeight = 0);

/**
 * @brief Converts a camera buffer to full-range YCbCr 4:2:0 for JPEG encoding,
 *        downscaled to @p outHeight rows like convertToRgb().
 *
 * NV12 (the Pi camera's default) is copied plane by plane with no colour
 * conversion; other frametypes go through RGB. Same validity rules as
 * convertToRgb().
 */
bool convertToYuv420(const camera_buffer_t* buffer, Yuv420Image& out, uint32_t outHeight = 0);

/**
 * @brief Switches the viewfinder to a frametype convertToRgb() handles, if it
 *        isn't using one already.
 */
int configureFrametype(camera_handle_t handle);

/**
 * @brief Switches the viewfinder to the smallest supported resolution that is
 *        at least @p height rows tall (the tallest one if none is), so less
 *        data has to be scaled down to @p height. 0 picks the tallest.
 */
int configureResolution(camera_handle_t handle, uint32_t height);
