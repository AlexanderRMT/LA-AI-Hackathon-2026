/**
 * @file jpeg_encoder.h
 *
 * @brief Small self-contained baseline JPEG encoder (YCbCr 4:2:0, standard
 *        Huffman tables), so the stream needs no libjpeg on the target.
 */

#pragma once

#include <cstdint>
#include <vector>

/** Planar full-range YCbCr 4:2:0 image: JPEG's native input format */
struct Yuv420Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> y;    // width * height
    std::vector<uint8_t> cb;   // chromaWidth() * chromaHeight()
    std::vector<uint8_t> cr;

    uint32_t chromaWidth() const { return (width + 1u) / 2u; }
    uint32_t chromaHeight() const { return (height + 1u) / 2u; }

    void resize(uint32_t w, uint32_t h)
    {
        width = w;
        height = h;
        y.resize(static_cast<size_t>(w) * h);
        cb.resize(static_cast<size_t>(chromaWidth()) * chromaHeight());
        cr.resize(cb.size());
    }
};

/** Converts packed RGB24 to full-range YCbCr 4:2:0 (chroma averaged over 2x2) */
void rgbToYuv420(const uint8_t* rgb, uint32_t width, uint32_t height, Yuv420Image& out);

class JpegEncoder {
public:
    /** @param quality 1 (smallest) to 100 (best), same scale as libjpeg */
    explicit JpegEncoder(int quality = 75);

    /**
     * @brief Encodes @p image into @p out (replacing its contents).
     *
     * With @p threads > 1 the image is split into horizontal strips separated
     * by JPEG restart markers, and the strips are encoded in parallel.
     */
    void encode(const Yuv420Image& image, std::vector<uint8_t>& out, unsigned threads = 1) const;

private:
    void encodeStrip(const Yuv420Image& image, uint32_t mcuRowBegin, uint32_t mcuRowEnd,
                     std::vector<uint8_t>& out) const;

    uint8_t lumaQ_[64];      // natural (row-major) order
    uint8_t chromaQ_[64];
    float lumaDiv_[64];      // reciprocal divisors folding in the AAN DCT scaling
    float chromaDiv_[64];
};
