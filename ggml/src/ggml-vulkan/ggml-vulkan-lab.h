// Engine-lab: GPU-friendly weight repacking for decode-time matvecs.
//
// Quantized weights are split into structure-of-arrays streams (same total byte count as the
// original layout) so that a matvec lane can use 16-byte aligned loads and decode scales cheaply.
// A tensor that was repacked at upload time is tracked in a registry; consumers that need the
// original layout (batched matmuls, readback) unpack it first.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct lab_fmt {
    ggml_type type;
    uint32_t  blk_bytes;   // bytes per block in the original layout
    uint32_t  blk_vals;    // values per block
    void (*repack)(const uint8_t * src, uint8_t * dst, size_t nblk);
    void (*unpack)(const uint8_t * src, uint8_t * dst, size_t nblk);
};

// ---- IQ4_XS: block = d(2) scales_h(2) scales_l(4) qs(128); 8 sub-blocks of 32 values, 16 qs bytes each.
// Original qs byte j of a sub-block holds value j (low nibble) and value j+16 (high nibble).
// Repacked: stream0 = nblk*128 B of qs with nibbles reordered so byte k holds values 2k (low) and 2k+1 (high);
//           stream1 = nblk*8 B of the original 8-byte header.
static inline void lab_repack_iq4_xs(const uint8_t * src, uint8_t * dst, size_t nblk) {
    uint8_t * qs_out   = dst;
    uint8_t * meta_out = dst + nblk * 128;
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t * blk = src + b * 136;
        memcpy(meta_out + b * 8, blk, 8);
        const uint8_t * qs = blk + 8;
        for (int ib = 0; ib < 8; ib++) {
            const uint8_t * o = qs + 16 * ib;
            uint8_t * n = qs_out + (b * 8 + ib) * 16;
            uint8_t v[32];
            for (int j = 0; j < 16; j++) { v[j] = o[j] & 0xf; v[j + 16] = o[j] >> 4; }
            for (int k = 0; k < 16; k++) { n[k] = v[2 * k] | (v[2 * k + 1] << 4); }
        }
    }
}
static inline void lab_unpack_iq4_xs(const uint8_t * src, uint8_t * dst, size_t nblk) {
    const uint8_t * qs_in   = src;
    const uint8_t * meta_in = src + nblk * 128;
    for (size_t b = 0; b < nblk; b++) {
        uint8_t * blk = dst + b * 136;
        memcpy(blk, meta_in + b * 8, 8);
        uint8_t * qs = blk + 8;
        for (int ib = 0; ib < 8; ib++) {
            const uint8_t * n = qs_in + (b * 8 + ib) * 16;
            uint8_t * o = qs + 16 * ib;
            uint8_t v[32];
            for (int k = 0; k < 16; k++) { v[2 * k] = n[k] & 0xf; v[2 * k + 1] = n[k] >> 4; }
            for (int j = 0; j < 16; j++) { o[j] = v[j] | (v[j + 16] << 4); }
        }
    }
}

static inline const lab_fmt * lab_get_fmt(ggml_type type) {
    static const lab_fmt fmts[] = {
        { GGML_TYPE_IQ4_XS, 136, 256, lab_repack_iq4_xs, lab_unpack_iq4_xs },
    };
    for (const auto & f : fmts) { if (f.type == type) return &f; }
    return nullptr;
}

static inline bool lab_repack_enabled() {
    static const bool en = [] { const char * e = getenv("LAB_REPACK"); return e && atoi(e) != 0; }();
    return en;
}
static inline bool lab_repack_all() {
    static const bool en = getenv("LAB_REPACK_ALL") != nullptr;
    return en;
}

// registry of tensors whose device storage is repacked
struct lab_registry {
    std::mutex mtx;
    std::unordered_set<const ggml_tensor *> repacked;
    struct pending { std::vector<uint8_t> data; size_t got = 0; };
    std::unordered_map<const ggml_tensor *, pending> staging;
    size_t max_nbytes = 0;
};
static inline lab_registry & lab_reg() { static lab_registry r; return r; }
static inline bool lab_is_repacked(const ggml_tensor * t) {
    if (!lab_repack_enabled()) { return false; }
    auto & r = lab_reg();
    std::lock_guard<std::mutex> g(r.mtx);
    return r.repacked.count(t) != 0;
}

// should this tensor be repacked on upload?
static inline bool lab_should_repack(const ggml_tensor * t) {
    if (!lab_repack_enabled()) { return false; }
    if (t->view_src != nullptr) { return false; }
    if (t->ne[2] != 1 || t->ne[3] != 1) { return false; }
    const lab_fmt * f = lab_get_fmt(t->type);
    if (!f) { return false; }
    if (t->ne[0] % f->blk_vals != 0) { return false; }
    if (!lab_repack_all()) {
        if (strncmp(t->name, "blk.", 4) != 0) { return false; }
        if (strstr(t->name, "exps") != nullptr) { return false; }
    }
    return true;
}
