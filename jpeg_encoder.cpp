/**
 * @file jpeg_encoder.cpp
 *
 * Baseline JPEG per ITU T.81, using the example tables from Annex K and the
 * AAN floating-point forward DCT (as in libjpeg's jfdctflt.c).
 */

#include "jpeg_encoder.h"

#include <cstdlib>
#include <thread>

namespace {

// Zigzag position -> natural (row-major) index
const uint8_t kZigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

// Annex K.1 quantization tables, natural order
const uint8_t kStdLumaQ[64] = {
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
};
const uint8_t kStdChromaQ[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
};

// Annex K.3 Huffman tables: code counts per length 1..16, then symbols
const uint8_t kDcLumaBits[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t kDcChromaBits[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t kDcVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

const uint8_t kAcLumaBits[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t kAcLumaVals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
};
const uint8_t kAcChromaBits[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const uint8_t kAcChromaVals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
};

// AAN DCT output scale factors: cos(k*pi/16) * sqrt(2), with k=0 -> 1
const float kAanScale[8] = {
    1.0f, 1.387039845f, 1.306562965f, 1.175875602f,
    1.0f, 0.785694958f, 0.541196100f, 0.275899379f,
};

struct HuffTable {
    uint16_t code[256] = {};
    uint8_t len[256] = {};

    HuffTable(const uint8_t* bits, const uint8_t* vals)
    {
        int k = 0;
        uint16_t c = 0;
        for (int l = 1; l <= 16; l++) {
            for (int i = 0; i < bits[l - 1]; i++) {
                code[vals[k]] = c++;
                len[vals[k]] = static_cast<uint8_t>(l);
                k++;
            }
            c = static_cast<uint16_t>(c << 1);
        }
    }
};

const HuffTable kDcLuma(kDcLumaBits, kDcVals);
const HuffTable kDcChroma(kDcChromaBits, kDcVals);
const HuffTable kAcLuma(kAcLumaBits, kAcLumaVals);
const HuffTable kAcChroma(kAcChromaBits, kAcChromaVals);

class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}

    void put(uint32_t bits, int len)
    {
        buffer_ = (buffer_ << len) | (bits & ((1u << len) - 1u));
        count_ += len;
        while (count_ >= 8) {
            uint8_t byte = static_cast<uint8_t>(buffer_ >> (count_ - 8));
            out_.push_back(byte);
            if (byte == 0xFF) {
                out_.push_back(0x00);   // byte stuffing
            }
            count_ -= 8;
        }
    }

    // Pads the last byte with 1-bits
    void flush()
    {
        int pad = (8 - count_ % 8) % 8;
        if (pad > 0) {
            put((1u << pad) - 1u, pad);
        }
    }

private:
    std::vector<uint8_t>& out_;
    uint32_t buffer_ = 0;
    int count_ = 0;
};

// One 8-point AAN forward DCT pass over d[0], d[s], ..., d[7s]
void fdct8(float* d, int s)
{
    float tmp0 = d[0 * s] + d[7 * s];
    float tmp7 = d[0 * s] - d[7 * s];
    float tmp1 = d[1 * s] + d[6 * s];
    float tmp6 = d[1 * s] - d[6 * s];
    float tmp2 = d[2 * s] + d[5 * s];
    float tmp5 = d[2 * s] - d[5 * s];
    float tmp3 = d[3 * s] + d[4 * s];
    float tmp4 = d[3 * s] - d[4 * s];

    // Even part
    float tmp10 = tmp0 + tmp3;
    float tmp13 = tmp0 - tmp3;
    float tmp11 = tmp1 + tmp2;
    float tmp12 = tmp1 - tmp2;
    d[0 * s] = tmp10 + tmp11;
    d[4 * s] = tmp10 - tmp11;
    float z1 = (tmp12 + tmp13) * 0.707106781f;
    d[2 * s] = tmp13 + z1;
    d[6 * s] = tmp13 - z1;

    // Odd part
    tmp10 = tmp4 + tmp5;
    tmp11 = tmp5 + tmp6;
    tmp12 = tmp6 + tmp7;
    float z5 = (tmp10 - tmp12) * 0.382683433f;
    float z2 = 0.541196100f * tmp10 + z5;
    float z4 = 1.306562965f * tmp12 + z5;
    float z3 = tmp11 * 0.707106781f;
    float z11 = tmp7 + z3;
    float z13 = tmp7 - z3;
    d[5 * s] = z13 + z2;
    d[3 * s] = z13 - z2;
    d[1 * s] = z11 + z4;
    d[7 * s] = z11 - z4;
}

inline int bitLength(int v)
{
    unsigned a = static_cast<unsigned>(std::abs(v));
    int n = 0;
    while (a != 0) {
        n++;
        a >>= 1;
    }
    return n;
}

// Magnitude bits for a coefficient: negatives are stored as v - 1 (ones' complement)
inline uint32_t magnitudeBits(int v, int cat)
{
    return static_cast<uint32_t>(v >= 0 ? v : v + (1 << cat) - 1);
}

// Transforms, quantizes and entropy-codes one 8x8 block (level-shifted samples)
void encodeBlock(float* block, const float* div, int& prevDc,
                 const HuffTable& dc, const HuffTable& ac, BitWriter& bw)
{
    for (int r = 0; r < 8; r++) {
        fdct8(block + 8 * r, 1);
    }
    for (int c = 0; c < 8; c++) {
        fdct8(block + c, 8);
    }

    int q[64];
    for (int i = 0; i < 64; i++) {
        int n = kZigzag[i];
        // Plain rounding: std::lrint is a slow libm call on QNX (~3x the whole encode)
        float t = block[n] * div[n];
        int v = static_cast<int>(t >= 0.0f ? t + 0.5f : t - 0.5f);
        q[i] = v < -1023 ? -1023 : (v > 1023 ? 1023 : v);   // baseline AC range
    }

    int diff = q[0] - prevDc;
    prevDc = q[0];
    int cat = bitLength(diff);
    bw.put(dc.code[cat], dc.len[cat]);
    if (cat > 0) {
        bw.put(magnitudeBits(diff, cat), cat);
    }

    int last = 63;
    while (last > 0 && q[last] == 0) {
        last--;
    }
    int run = 0;
    for (int i = 1; i <= last; i++) {
        if (q[i] == 0) {
            run++;
            continue;
        }
        while (run >= 16) {
            bw.put(ac.code[0xF0], ac.len[0xF0]);   // ZRL: 16 zeros
            run -= 16;
        }
        cat = bitLength(q[i]);
        int sym = (run << 4) | cat;
        bw.put(ac.code[sym], ac.len[sym]);
        bw.put(magnitudeBits(q[i], cat), cat);
        run = 0;
    }
    if (last < 63) {
        bw.put(ac.code[0x00], ac.len[0x00]);   // EOB
    }
}

// Rounds and clamps to 0-255 (saturated blue/red reach 255.5 before rounding)
inline uint8_t toByte(float v)
{
    return static_cast<uint8_t>(v <= 0.0f ? 0 : (v >= 255.0f ? 255 : static_cast<int>(v + 0.5f)));
}

void putMarker(std::vector<uint8_t>& out, uint8_t marker, uint16_t length)
{
    out.push_back(0xFF);
    out.push_back(marker);
    out.push_back(static_cast<uint8_t>(length >> 8));
    out.push_back(static_cast<uint8_t>(length & 0xFF));
}

void put16(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void putHuffTable(std::vector<uint8_t>& out, uint8_t classId, const uint8_t* bits, const uint8_t* vals, int numVals)
{
    out.push_back(classId);
    out.insert(out.end(), bits, bits + 16);
    out.insert(out.end(), vals, vals + numVals);
}

}  // namespace

JpegEncoder::JpegEncoder(int quality)
{
    // libjpeg quality scaling
    if (quality < 1) {
        quality = 1;
    } else if (quality > 100) {
        quality = 100;
    }
    int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
    for (int n = 0; n < 64; n++) {
        int l = (kStdLumaQ[n] * scale + 50) / 100;
        int c = (kStdChromaQ[n] * scale + 50) / 100;
        lumaQ_[n] = static_cast<uint8_t>(l < 1 ? 1 : (l > 255 ? 255 : l));
        chromaQ_[n] = static_cast<uint8_t>(c < 1 ? 1 : (c > 255 ? 255 : c));

        float aan = kAanScale[n / 8] * kAanScale[n % 8] * 8.0f;
        lumaDiv_[n] = 1.0f / (lumaQ_[n] * aan);
        chromaDiv_[n] = 1.0f / (chromaQ_[n] * aan);
    }
}

void rgbToYuv420(const uint8_t* rgb, uint32_t width, uint32_t height, Yuv420Image& out)
{
    out.resize(width, height);
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t* px = rgb + static_cast<size_t>(y) * width * 3u;
        uint8_t* dst = &out.y[static_cast<size_t>(y) * width];
        for (uint32_t x = 0; x < width; x++, px += 3) {
            dst[x] = toByte(0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2]);
        }
    }
    const uint32_t cw = out.chromaWidth();
    for (uint32_t cy = 0; cy < out.chromaHeight(); cy++) {
        for (uint32_t cx = 0; cx < cw; cx++) {
            float cb = 0.0f;
            float cr = 0.0f;
            for (uint32_t dy = 0; dy < 2; dy++) {
                uint32_t sy = (2 * cy + dy < height) ? 2 * cy + dy : height - 1;
                for (uint32_t dx = 0; dx < 2; dx++) {
                    uint32_t sx = (2 * cx + dx < width) ? 2 * cx + dx : width - 1;
                    const uint8_t* p = rgb + (static_cast<size_t>(sy) * width + sx) * 3u;
                    cb += -0.168736f * p[0] - 0.331264f * p[1] + 0.5f * p[2];
                    cr += 0.5f * p[0] - 0.418688f * p[1] - 0.081312f * p[2];
                }
            }
            size_t i = static_cast<size_t>(cy) * cw + cx;
            out.cb[i] = toByte(cb * 0.25f + 128.0f);
            out.cr[i] = toByte(cr * 0.25f + 128.0f);
        }
    }
}

void JpegEncoder::encode(const Yuv420Image& image, std::vector<uint8_t>& out, unsigned threads) const
{
    out.clear();
    const uint32_t width = image.width;
    const uint32_t height = image.height;
    if (width == 0 || height == 0 || width > 65535 || height > 65535) {
        return;
    }
    out.reserve(static_cast<size_t>(width) * height / 4u + 1024u);

    // Split into strips of whole MCU rows; each strip is one restart interval
    const uint32_t mcuCols = (width + 15u) / 16u;
    const uint32_t mcuRows = (height + 15u) / 16u;
    uint32_t strips = (threads < 1u) ? 1u : threads;
    if (strips > mcuRows) {
        strips = mcuRows;
    }
    uint32_t rowsPerStrip = (mcuRows + strips - 1u) / strips;
    if (static_cast<uint64_t>(rowsPerStrip) * mcuCols > 65535u) {
        rowsPerStrip = mcuRows;   // restart interval would overflow; encode in one piece
    }
    strips = (mcuRows + rowsPerStrip - 1u) / rowsPerStrip;

    // SOI + JFIF APP0
    out.push_back(0xFF);
    out.push_back(0xD8);
    putMarker(out, 0xE0, 16);
    const uint8_t jfif[] = {'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0};
    out.insert(out.end(), jfif, jfif + sizeof(jfif));

    // DQT: tables are written in zigzag order
    putMarker(out, 0xDB, 2 + 2 * 65);
    out.push_back(0x00);
    for (int i = 0; i < 64; i++) {
        out.push_back(lumaQ_[kZigzag[i]]);
    }
    out.push_back(0x01);
    for (int i = 0; i < 64; i++) {
        out.push_back(chromaQ_[kZigzag[i]]);
    }

    // SOF0: 8-bit, 3 components, Y sampled 2x2, Cb/Cr 1x1 (4:2:0)
    putMarker(out, 0xC0, 17);
    out.push_back(8);
    put16(out, height);
    put16(out, width);
    out.push_back(3);
    const uint8_t components[] = {1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1};
    out.insert(out.end(), components, components + sizeof(components));

    // DHT
    putMarker(out, 0xC4, 2 + 2 * (17 + 12) + 2 * (17 + 162));
    putHuffTable(out, 0x00, kDcLumaBits, kDcVals, 12);
    putHuffTable(out, 0x10, kAcLumaBits, kAcLumaVals, 162);
    putHuffTable(out, 0x01, kDcChromaBits, kDcVals, 12);
    putHuffTable(out, 0x11, kAcChromaBits, kAcChromaVals, 162);

    // DRI: restart interval in MCUs
    if (strips > 1u) {
        putMarker(out, 0xDD, 4);
        put16(out, rowsPerStrip * mcuCols);
    }

    // SOS
    putMarker(out, 0xDA, 12);
    const uint8_t scan[] = {3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0};
    out.insert(out.end(), scan, scan + sizeof(scan));

    if (strips == 1u) {
        encodeStrip(image, 0, mcuRows, out);
    } else {
        std::vector<std::vector<uint8_t>> parts(strips);
        std::vector<std::thread> workers;
        workers.reserve(strips - 1u);
        for (uint32_t i = 1; i < strips; i++) {
            uint32_t begin = i * rowsPerStrip;
            uint32_t end = (begin + rowsPerStrip < mcuRows) ? begin + rowsPerStrip : mcuRows;
            workers.emplace_back([this, &image, &parts, i, begin, end] {
                encodeStrip(image, begin, end, parts[i]);
            });
        }
        encodeStrip(image, 0, rowsPerStrip, parts[0]);
        for (std::thread& t : workers) {
            t.join();
        }
        for (uint32_t i = 0; i < strips; i++) {
            out.insert(out.end(), parts[i].begin(), parts[i].end());
            if (i + 1u < strips) {
                out.push_back(0xFF);
                out.push_back(static_cast<uint8_t>(0xD0 + (i & 7u)));   // RSTn
            }
        }
    }

    // EOI
    out.push_back(0xFF);
    out.push_back(0xD9);
}

// Entropy-codes MCU rows [mcuRowBegin, mcuRowEnd) as one restart interval:
// DC predictors start at zero and the final byte is padded.
void JpegEncoder::encodeStrip(const Yuv420Image& image, uint32_t mcuRowBegin, uint32_t mcuRowEnd,
                              std::vector<uint8_t>& out) const
{
    const uint32_t width = image.width;
    const uint32_t height = image.height;
    const uint32_t cw = image.chromaWidth();
    const uint32_t ch = image.chromaHeight();

    BitWriter bw(out);
    int prevY = 0, prevCb = 0, prevCr = 0;
    float y16[256];
    float block[64];

    for (uint32_t row = mcuRowBegin; row < mcuRowEnd; row++) {
        const uint32_t my = row * 16u;
        for (uint32_t mx = 0; mx < width; mx += 16) {
            // Load the 16x16 luma area, level-shifted (edges repeat the last row/column)
            for (uint32_t r = 0; r < 16; r++) {
                uint32_t sy = (my + r < height) ? my + r : height - 1;
                const uint8_t* line = &image.y[static_cast<size_t>(sy) * width];
                if (mx + 16u <= width) {
                    for (uint32_t c = 0; c < 16; c++) {
                        y16[r * 16 + c] = line[mx + c] - 128.0f;
                    }
                } else {
                    for (uint32_t c = 0; c < 16; c++) {
                        uint32_t sx = (mx + c < width) ? mx + c : width - 1;
                        y16[r * 16 + c] = line[sx] - 128.0f;
                    }
                }
            }
            for (int by = 0; by < 2; by++) {
                for (int bx = 0; bx < 2; bx++) {
                    for (int r = 0; r < 8; r++) {
                        for (int c = 0; c < 8; c++) {
                            block[r * 8 + c] = y16[(by * 8 + r) * 16 + bx * 8 + c];
                        }
                    }
                    encodeBlock(block, lumaDiv_, prevY, kDcLuma, kAcLuma, bw);
                }
            }

            // Chroma is already subsampled: one 8x8 block per plane
            const uint32_t cmx = mx / 2u;
            const uint32_t cmy = my / 2u;
            const uint8_t* planes[2] = {image.cb.data(), image.cr.data()};
            int* prev[2] = {&prevCb, &prevCr};
            for (int p = 0; p < 2; p++) {
                for (uint32_t r = 0; r < 8; r++) {
                    uint32_t cy = (cmy + r < ch) ? cmy + r : ch - 1;
                    const uint8_t* line = planes[p] + static_cast<size_t>(cy) * cw;
                    for (uint32_t c = 0; c < 8; c++) {
                        uint32_t cx = (cmx + c < cw) ? cmx + c : cw - 1;
                        block[r * 8 + c] = line[cx] - 128.0f;
                    }
                }
                encodeBlock(block, chromaDiv_, *prev[p], kDcChroma, kAcChroma, bw);
            }
        }
    }
    bw.flush();
}
