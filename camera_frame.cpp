/**
 * @file camera_frame.cpp
 *
 * Frame conversion is based on camera_example1_callback (Apache-2.0,
 * BlackBerry Limited).
 */

#include "camera_frame.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>

namespace {

// Frametypes convertToRgb() handles, in order of preference
const camera_frametype_t kSupportedFrametypes[] = {
    CAMERA_FRAMETYPE_RGB8888,
    CAMERA_FRAMETYPE_BGR8888,
    CAMERA_FRAMETYPE_YCBYCR,
    CAMERA_FRAMETYPE_CBYCRY,
    CAMERA_FRAMETYPE_NV12,
};

inline uint8_t clamp8(int v)
{
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// BT.601 YCbCr -> RGB (integer approximation)
inline void yuvToRgb(int y, int cb, int cr, uint8_t* out)
{
    int c = y - 16;
    int d = cb - 128;
    int e = cr - 128;
    out[0] = clamp8((298 * c + 409 * e + 128) >> 8);
    out[1] = clamp8((298 * c - 100 * d - 208 * e + 128) >> 8);
    out[2] = clamp8((298 * c + 516 * d + 128) >> 8);
}

// Output size for scaling a srcW x srcH frame down to outHeight rows, keeping
// the aspect ratio; the width is rounded to even for 4:2:0 chroma
void scaledSize(uint32_t srcW, uint32_t srcH, uint32_t outHeight, uint32_t& w, uint32_t& h)
{
    if (outHeight == 0 || outHeight >= srcH) {
        w = srcW;
        h = srcH;
        return;
    }
    h = outHeight;
    w = static_cast<uint32_t>((static_cast<uint64_t>(srcW) * outHeight + srcH) / (2u * srcH) * 2u);
}

// Nearest-neighbour source index for each of n output samples spanning src,
// sampling at sample centres (identity when n == src). Reversed, the output
// runs back to front, so reversing both axes rotates the picture 180 degrees.
void sampleMap(uint32_t n, uint32_t src, std::vector<uint32_t>& map, bool reversed)
{
    map.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        map[i] = static_cast<uint32_t>((2u * static_cast<uint64_t>(i) + 1u) * src / (2u * static_cast<uint64_t>(n)));
    }
    if (reversed) {
        std::reverse(map.begin(), map.end());
    }
}

}  // namespace

bool convertToRgb(const camera_buffer_t* buffer, Frame& out, uint32_t outHeight, bool rotate180)
{
    const uint8_t* base = buffer->framebuf;

    uint32_t srcW;
    uint32_t srcH;
    uint32_t stride;
    switch (buffer->frametype) {
    case CAMERA_FRAMETYPE_RGB8888:
        srcW = buffer->framedesc.rgb8888.width;
        srcH = buffer->framedesc.rgb8888.height;
        stride = buffer->framedesc.rgb8888.stride;
        break;
    case CAMERA_FRAMETYPE_BGR8888:
        srcW = buffer->framedesc.bgr8888.width;
        srcH = buffer->framedesc.bgr8888.height;
        stride = buffer->framedesc.bgr8888.stride;
        break;
    case CAMERA_FRAMETYPE_YCBYCR:
        srcW = buffer->framedesc.ycbycr.width;
        srcH = buffer->framedesc.ycbycr.height;
        stride = buffer->framedesc.ycbycr.stride;
        break;
    case CAMERA_FRAMETYPE_CBYCRY:
        srcW = buffer->framedesc.cbycry.width;
        srcH = buffer->framedesc.cbycry.height;
        stride = buffer->framedesc.cbycry.stride;
        break;
    case CAMERA_FRAMETYPE_NV12:
        srcW = buffer->framedesc.nv12.width;
        srcH = buffer->framedesc.nv12.height;
        stride = buffer->framedesc.nv12.stride;
        break;
    default:
        return false;
    }

    uint32_t w;
    uint32_t h;
    scaledSize(srcW, srcH, outHeight, w, h);
    thread_local std::vector<uint32_t> xs;
    thread_local std::vector<uint32_t> ys;
    sampleMap(w, srcW, xs, rotate180);
    sampleMap(h, srcH, ys, rotate180);
    out.width = w;
    out.height = h;
    out.rgb.resize(static_cast<size_t>(w) * h * 3u);

    switch (buffer->frametype) {
    case CAMERA_FRAMETYPE_RGB8888:
    case CAMERA_FRAMETYPE_BGR8888:
    {
        // Byte order matches camera_example1_callback: RGB8888 is R,G,B,A and
        // BGR8888 is B,G,R,A in memory.
        bool isRgb = (buffer->frametype == CAMERA_FRAMETYPE_RGB8888);
        int r = isRgb ? 0 : 2;
        int b = isRgb ? 2 : 0;
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t* line = base + static_cast<size_t>(ys[y]) * stride;
            uint8_t* dst = &out.rgb[static_cast<size_t>(y) * w * 3u];
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t* px = line + 4u * xs[x];
                dst[3 * x + 0] = px[r];
                dst[3 * x + 1] = px[1];
                dst[3 * x + 2] = px[b];
            }
        }
        return true;
    }
    case CAMERA_FRAMETYPE_YCBYCR:
    case CAMERA_FRAMETYPE_CBYCRY:
    {
        // Each 4-byte pair covers two pixels.
        // YCBYCR: Y0 Cb Y1 Cr    CBYCRY: Cb Y0 Cr Y1
        bool yFirst = (buffer->frametype == CAMERA_FRAMETYPE_YCBYCR);
        int y0 = yFirst ? 0 : 1, cb = yFirst ? 1 : 0, y1 = yFirst ? 2 : 3, cr = yFirst ? 3 : 2;
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t* line = base + static_cast<size_t>(ys[y]) * stride;
            uint8_t* dst = &out.rgb[static_cast<size_t>(y) * w * 3u];
            for (uint32_t x = 0; x < w; x++) {
                uint32_t sx = xs[x];
                const uint8_t* pair = line + 2u * (sx & ~1u);
                yuvToRgb(pair[(sx & 1u) ? y1 : y0], pair[cb], pair[cr], dst + 3 * x);
            }
        }
        return true;
    }
    case CAMERA_FRAMETYPE_NV12:
    {
        const uint8_t* uvBase = base + buffer->framedesc.nv12.uv_offset;
        int64_t uvStride = buffer->framedesc.nv12.uv_stride;
        for (uint32_t y = 0; y < h; y++) {
            uint32_t sy = ys[y];
            const uint8_t* yLine = base + static_cast<size_t>(sy) * stride;
            const uint8_t* uvLine = uvBase + static_cast<int64_t>(sy / 2u) * uvStride;
            uint8_t* dst = &out.rgb[static_cast<size_t>(y) * w * 3u];
            for (uint32_t x = 0; x < w; x++) {
                uint32_t sx = xs[x];
                const uint8_t* uv = uvLine + (sx & ~1u);
                yuvToRgb(yLine[sx], uv[0], uv[1], dst + 3 * x);
            }
        }
        return true;
    }
    default:
        return false;
    }
}

bool convertToYuv420(const camera_buffer_t* buffer, Yuv420Image& out, uint32_t outHeight, bool rotate180)
{
    if (buffer->frametype != CAMERA_FRAMETYPE_NV12) {
        thread_local Frame rgb;
        if (!convertToRgb(buffer, rgb, outHeight, rotate180)) {
            return false;
        }
        rgbToYuv420(rgb.rgb.data(), rgb.width, rgb.height, out);
        return true;
    }

    // Camera NV12 is video range (Y 16-235, C 16-240); JPEG expects full range.
    // Same scaling as yuvToRgb() above.
    static const struct RangeLut {
        uint8_t y[256];
        uint8_t c[256];
        RangeLut()
        {
            for (int v = 0; v < 256; v++) {
                y[v] = clamp8(((v - 16) * 255 * 2 + 219) / (219 * 2));
                int c2 = (v - 128) * 255;
                c[v] = clamp8(128 + (c2 >= 0 ? (c2 + 112) / 224 : (c2 - 112) / 224));
            }
        }
    } lut;

    const auto& d = buffer->framedesc.nv12;
    const uint32_t srcCw = d.width / 2u;
    const uint32_t srcCh = d.height / 2u;
    uint32_t w;
    uint32_t h;
    scaledSize(d.width, d.height, outHeight, w, h);
    out.resize(w, h);
    thread_local std::vector<uint32_t> xs;
    thread_local std::vector<uint32_t> ys;
    sampleMap(w, d.width, xs, rotate180);
    sampleMap(h, d.height, ys, rotate180);
    const uint8_t* base = buffer->framebuf;

    for (uint32_t y = 0; y < h; y++) {
        const uint8_t* src = base + static_cast<size_t>(ys[y]) * d.stride;
        uint8_t* dst = &out.y[static_cast<size_t>(y) * w];
        for (uint32_t x = 0; x < w; x++) {
            dst[x] = lut.y[src[xs[x]]];
        }
    }

    // Output chroma sample (cx, cy) sits over output luma (2cx, 2cy), which
    // samples source luma (xs[2cx], ys[2cy]), i.e. source chroma at half that
    const uint32_t cw = out.chromaWidth();
    const uint8_t* uvBase = base + d.uv_offset;
    for (uint32_t cy = 0; cy < out.chromaHeight(); cy++) {
        uint32_t sy = ys[(2u * cy < h) ? 2u * cy : h - 1] / 2u;
        sy = (sy < srcCh) ? sy : srcCh - 1;
        const uint8_t* uvLine = uvBase + static_cast<int64_t>(sy) * d.uv_stride;
        uint8_t* cb = &out.cb[static_cast<size_t>(cy) * cw];
        uint8_t* cr = &out.cr[static_cast<size_t>(cy) * cw];
        for (uint32_t cx = 0; cx < cw; cx++) {
            uint32_t sx = xs[(2u * cx < w) ? 2u * cx : w - 1] / 2u;
            sx = (sx < srcCw) ? sx : srcCw - 1;
            cb[cx] = lut.c[uvLine[2 * sx]];
            cr[cx] = lut.c[uvLine[2 * sx + 1]];
        }
    }
    return true;
}

int configureFrametype(camera_handle_t handle)
{
    camera_frametype_t current = CAMERA_FRAMETYPE_UNSPECIFIED;
    int err = camera_get_vf_property(handle, CAMERA_IMGPROP_FORMAT, &current);
    if (err != CAMERA_EOK) {
        std::printf("Failed to get frametype: err = %d\n", err);
        return err;
    }
    for (camera_frametype_t t : kSupportedFrametypes) {
        if (t == current) {
            return CAMERA_EOK;
        }
    }

    uint32_t numSupported = 0;
    err = camera_get_supported_vf_frame_types(handle, 0, &numSupported, nullptr);
    if (err != CAMERA_EOK) {
        std::printf("Failed to get number of supported frametypes: err = %d\n", err);
        return err;
    }
    std::vector<camera_frametype_t> available(numSupported);
    err = camera_get_supported_vf_frame_types(handle, numSupported, &numSupported, available.data());
    if (err != CAMERA_EOK) {
        std::printf("Failed to get supported frametypes: err = %d\n", err);
        return err;
    }
    for (camera_frametype_t want : kSupportedFrametypes) {
        for (uint32_t i = 0; i < numSupported; i++) {
            if (available[i] == want) {
                std::printf("Switching frametype %d -> %d\n", (int)current, (int)want);
                return camera_set_vf_property(handle, CAMERA_IMGPROP_FORMAT, want);
            }
        }
    }
    std::printf("Camera offers no frametype this tool can convert\n");
    return ENOTSUP;
}

int configureResolution(camera_handle_t handle, uint32_t height)
{
    uint32_t numSupported = 0;
    int err = camera_get_supported_vf_resolutions(handle, 0, &numSupported, nullptr);
    if (err != CAMERA_EOK) {
        std::printf("Failed to get number of supported resolutions: err = %d\n", err);
        return err;
    }
    std::vector<camera_res_t> available(numSupported);
    err = camera_get_supported_vf_resolutions(handle, numSupported, &numSupported, available.data());
    if (err != CAMERA_EOK || numSupported == 0) {
        std::printf("Failed to get supported resolutions: err = %d\n", err);
        return (err != CAMERA_EOK) ? err : ENOTSUP;
    }

    // Smallest one at least `height` tall, else the tallest
    const camera_res_t* best = &available[0];
    for (uint32_t i = 1; i < numSupported; i++) {
        const camera_res_t& r = available[i];
        bool fits = (height != 0 && r.height >= height);
        bool bestFits = (height != 0 && best->height >= height);
        if (fits ? (!bestFits || r.height < best->height) : (!bestFits && r.height > best->height)) {
            best = &r;
        }
    }

    uint32_t curW = 0;
    uint32_t curH = 0;
    camera_get_vf_property(handle, CAMERA_IMGPROP_WIDTH, &curW, CAMERA_IMGPROP_HEIGHT, &curH);
    if (curW == best->width && curH == best->height) {
        return CAMERA_EOK;
    }
    std::printf("Switching viewfinder %ux%u -> %ux%u\n", curW, curH, best->width, best->height);
    err = camera_set_vf_property(handle, CAMERA_IMGPROP_WIDTH, best->width, CAMERA_IMGPROP_HEIGHT, best->height);
    if (err != CAMERA_EOK) {
        std::printf("Failed to set resolution: err = %d\n", err);
    }
    return err;
}
