/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// Shared static contract/configuration only. No tensor contents are read here.
// CANN and Torch may use different libstdc++ ABIs. Keep these private helpers
// local to each translation unit so their destructors cannot interpose.
namespace {
namespace qli {
enum Dtype { U8, F32, I32, BF16, Invalid };
struct TensorSpec {
    std::vector<int64_t> dims, strides;
    Dtype dtype = Invalid;
    bool present = false;
};
struct Attrs {
    int64_t topk, quant_mode, max_q, mask, ratio, candidates, block;
    std::string layout_q, layout_k;
    bool values;
};
struct Config {
    int64_t rows = 0;
    int64_t batches = 0;
    int64_t max_q = 0;
    int64_t max_k = 0;
    int64_t page_size = 0;
    int64_t k_rows = 0;
    int64_t s_rows = 0;
    int64_t k_page_stride = 0;
    int64_t ks_page_stride = 0;
    int64_t k_stride = 0;
    int64_t ks_stride = 0;
    int64_t table_cols = 0;
    int64_t cube = 0;
    int64_t vector = 0;
    int64_t plan_elems = 0;
    int64_t capacity = 0;
    int64_t flags = 0;
    int64_t cmp_ratio = 0;
    int64_t ws_scores = 0;
    int64_t ws_index_map = 0;
    int64_t ws_partial = 0;
    int64_t score_cols = 0;
    int64_t aux_cols = 0;
    int64_t topk = 0;
    int64_t tile_n = 128, select = 0, shard = 4096, fixed = 0, ratio_mode = 0;
    int64_t parts = 0, split = 0, schedule = 0, workspace_bytes = 0;
    bool fast = false;
};
// Counts are nonnegative. Zero denotes an unavailable or invalid divisor.
inline int64_t Ceil(int64_t count, int64_t divisor) {
    if (divisor <= 0) return 0;
    const auto result = std::lldiv(count, divisor);
    return result.quot + (result.rem != 0);
}
inline int64_t Align(int64_t n) { return Ceil(n, 512) * 512; }
inline bool Shape(const TensorSpec &s, std::initializer_list<int64_t> dims) {
    return s.present && s.dims == std::vector<int64_t>(dims);
}
inline bool Dense(const TensorSpec &s) {
    if (!s.present || s.strides.size() != s.dims.size()) return false;
    int64_t stride = 1;
    for (size_t i=s.dims.size(); i>0; --i) {
        if (s.dims[i-1] != 1 && s.strides[i-1] != stride) return false;
        stride *= s.dims[i-1];
    }
    return true;
}
inline bool InnerDense(const TensorSpec &s) {
    if (!s.present || s.dims.empty() || s.strides.size()!=s.dims.size()) return false;
    int64_t stride = 1;
    for (size_t i=s.dims.size(); i>1; --i) {
        if (s.strides[i-1]!=stride) return false;
        stride *= s.dims[i-1];
    }
    return s.strides[0]>=stride;
}
inline int64_t Capacity(int64_t n, std::initializer_list<int64_t> values) {
    for (auto v: values) if (v>=n) return v;
    return 0;
}
using InputSpecs = std::array<TensorSpec, 12>;

inline bool Reject(std::string &error, const char *reason) {
    error = reason;
    return false;
}

inline bool ValidateAttrs(const Attrs &a, std::string &error) {
    if (a.quant_mode!=1 || a.layout_q!="TND" ||
        (a.layout_k!="TND" && a.layout_k!="PA_BBND")) return Reject(error, "invalid quantization/layout");
    if (a.topk<1 || a.topk>8192 || (a.mask!=0 && a.mask!=3) ||
        a.ratio<1 || a.ratio>128) return Reject(error, "invalid topk/mask/ratio");
    if ((a.candidates==-1 && a.block!=-1) ||
        (a.candidates!=-1 && (a.candidates<=0 || a.block!=8))) return Reject(error, "invalid candidate attributes");
    return true;
}

inline bool ReadQuery(const InputSpecs &x, Config &c, std::string &error) {
    if (!x[0].present || x[0].dims.size()!=3 || x[0].dims[0]<=0 ||
        !Shape(x[0],{x[0].dims[0],32,64}) || x[0].dtype!=U8 || !Dense(x[0])) return Reject(error, "invalid q");
    c.rows=x[0].dims[0];
    if (!Shape(x[2],{c.rows,32}) || x[2].dtype!=F32 || !Dense(x[2])) return Reject(error, "invalid weights");
    if (!Shape(x[3],{c.rows,32,2,2}) || x[3].dtype!=U8 || !Dense(x[3])) return Reject(error, "invalid query scales");
    return true;
}

inline bool ReadSequence(const InputSpecs &x, const Attrs &a, Config &c, std::string &error) {
    c.batches=1;
    if (x[5].present) {
        if (x[5].dims.size()!=1 || x[5].dims[0]<2) return Reject(error, "invalid query prefixes");
        c.batches=x[5].dims[0]-1;
    }
    for (size_t i=5;i<=9;++i) {
        if (!x[i].present) continue;
        if (!Shape(x[i],{c.batches+(i<=6 ? 1 : 0)}) || x[i].dtype!=I32 || !Dense(x[i])) return Reject(error, "invalid sequence tensor");
    }
    if ((a.mask==3 && a.ratio!=1)!=x[9].present) return Reject(error, "compression residual presence mismatch");
    if (x[11].present && (!Shape(x[11],{c.rows,1}) || x[11].dtype!=I32 || !Dense(x[11]))) return Reject(error, "invalid index offsets");
    return true;
}

inline bool ReadPagedKey(const InputSpecs &x, Config &c, std::string &error) {
    if (x[1].dims.size()!=4 || x[1].dims[2]!=1 || x[1].dims[3]!=64) return Reject(error, "invalid PA key shape");
    c.page_size=x[1].dims[1];
    if (c.page_size<=0 || c.page_size>1024 || c.page_size%16) return Reject(error, "invalid page size");
    if (!Shape(x[4],{x[1].dims[0],c.page_size,1,2,2}) ||
        x[1].strides[0]%64 || x[4].strides[0]%4) return Reject(error, "invalid PA scales/stride");
    if (x[10].present && (x[10].dims.size()!=2 || x[10].dims[0]!=c.batches ||
        x[10].dims[1]<=0 || x[10].dtype!=I32 || !Dense(x[10]))) return Reject(error, "invalid page table");
    if (!x[10].present && c.batches!=1) return Reject(error, "multi-batch PA needs table");
    c.table_cols=x[10].present ? x[10].dims[1] : x[1].dims[0];
    if (c.table_cols > std::numeric_limits<int64_t>::max() / c.page_size)
        return Reject(error, "page table extent overflows");
    c.max_k=c.table_cols*c.page_size;
    c.tile_n=std::gcd(c.page_size,int64_t(128));
    c.k_page_stride=x[1].strides[0]/64;
    c.ks_page_stride=x[4].strides[0]/4;
    c.k_rows=(x[1].dims[0]-1)*c.k_page_stride+c.page_size;
    c.s_rows=(x[4].dims[0]-1)*c.ks_page_stride+c.page_size;
    c.k_stride=64; c.ks_stride=4;
    return true;
}

inline bool ReadDenseKey(const InputSpecs &x, Config &c, std::string &error) {
    if (!Shape(x[1],{x[1].dims[0],1,64}) ||
        !Shape(x[4],{x[1].dims[0],1,2,2})) return Reject(error, "invalid TND key/scales");
    if (x[10].present || (c.batches>1 && !x[6].present)) return Reject(error, "TND needs prefixes and no page table");
    c.max_k=x[1].dims[0]; c.page_size=128; c.table_cols=1;
    c.k_rows=c.max_k; c.s_rows=c.max_k;
    c.k_stride=x[1].strides[0]; c.ks_stride=x[4].strides[0];
    return true;
}

inline bool ReadKey(const InputSpecs &x, const Attrs &a, Config &c, std::string &error) {
    if (!x[1].present || x[1].dims.empty() || x[1].dims[0]<=0 ||
        !x[4].present || x[1].dtype!=U8 || x[4].dtype!=U8 ||
        !InnerDense(x[1]) || !InnerDense(x[4])) return Reject(error, "invalid key storage/strides");
    return a.layout_k == "PA_BBND" ? ReadPagedKey(x, c, error) : ReadDenseKey(x, c, error);
}

inline bool UseFastSchedule(const InputSpecs &x, const Attrs &a, const Config &c, int64_t actual_shard) {
    return a.layout_k == "PA_BBND" && c.page_size==128 && c.max_k>=8192 && c.max_k<=131072 &&
        c.max_k%256==0 && a.topk==512 && a.candidates==-1 && x[10].present &&
        x[5].present && x[7].present && x[8].present && c.parts*512<=actual_shard;
}

inline void ChooseShards(Config &c, int64_t cubes, int64_t groups, int64_t shard_tiles) {
    if (c.fast && c.max_k==131072 && cubes==28) {
        // Match the JIT geometry selection. AICPU assigns the actual tasks.
        const int64_t fine_tiles=Ceil(c.max_k,14*256);
        const int64_t coarse=Ceil(groups*c.parts,cubes)*shard_tiles;
        const int64_t fine=Ceil(groups*14,cubes)*fine_tiles;
        if (fine*5<=coarse*4) {
            c.parts=14;
            c.shard=fine_tiles*256;
        }
    }
}

inline void SetSchedule(const InputSpecs &x, Config &c, bool paged, bool fused, bool post, bool allow_empty) {
    c.flags=int64_t(paged && !c.fast) | (int64_t(x[10].present)<<1) |
        (int64_t(x[11].present)<<2) | (int64_t(fused)<<3) | (int64_t(post)<<4) |
        (int64_t(allow_empty)<<5) | (int64_t(c.page_size==c.tile_n)<<6);
    int64_t plan_flags=int64_t(c.fast) | (int64_t(x[5].present)<<1) |
        (int64_t(x[6].present)<<2) | (int64_t(x[7].present)<<3) |
        (int64_t(x[8].present)<<4) | (int64_t(x[9].present)<<5) | (int64_t(allow_empty)<<6);
    c.schedule=c.split | ((c.fast?256:c.tile_n)<<8) | (c.cube<<24) |
        (c.vector<<32) | (plan_flags<<40);
}

inline void SetWorkspace(Config &c, const Attrs &a, int64_t select_width, int64_t actual_shard, bool post) {
    if (c.fast) {
        c.tile_n=128;
        c.select=0;
        if (c.parts!=14) c.shard=Capacity(actual_shard,{4096,8192,16384});
    }
    // This profile bounds both token and fused-candidate output tiles at 512.
    c.fixed=select_width && a.topk==512 && a.candidates<=512 ? 512 : 0;
    c.ratio_mode=a.mask==3 && a.ratio==2 ? 2 : 0;
    c.capacity=a.candidates; c.cmp_ratio=a.ratio; c.topk=a.topk;
    c.score_cols=c.fast ? c.parts*512 : c.max_k;
    c.aux_cols=c.fast ? c.score_cols : Ceil(c.max_k,8);
    c.ws_scores=0; c.ws_index_map=Align(c.rows*c.score_cols*2);
    c.ws_partial=c.ws_index_map+(c.fast ? Align(c.rows*c.score_cols*4) : 512);
    c.workspace_bytes=c.ws_partial+(post ? Align(c.rows*c.aux_cols*2) : 512);
    if (c.fast) c.ws_partial=0;
}

inline void ConfigureSelection(const InputSpecs &x, const Attrs &a, Config &c, int64_t groups,
                               int64_t actual_shard, int64_t cubes, int64_t per_cube) {
    const int64_t select_width=(!c.fast && c.max_k<=16384) ? Ceil(c.max_k,256)*256 : 0;
    c.select=select_width ? Capacity(select_width,{256,512,1024,1536,2048,4096,8192,16384}) : 0;
    const bool fused=select_width && a.candidates>0 && a.candidates<=8192;
    const bool post=a.candidates>0 && !fused;
    const bool allow_empty=select_width && (a.candidates<0 || fused);
    c.split=c.fast ? c.parts : std::min(int64_t(7),Ceil(c.max_k,c.tile_n));
    const int64_t items=groups*c.split;
    c.cube=std::min(cubes,std::max(items,Ceil(c.rows,per_cube)));
    c.vector=c.cube*per_cube;
    c.plan_elems=1+3*c.rows+2*c.vector+2*c.cube+8*items;
    SetSchedule(x, c, a.layout_k == "PA_BBND", fused, post, allow_empty);
    SetWorkspace(c, a, select_width, actual_shard, post);
}

inline bool Derive(const InputSpecs &x, const Attrs &a, int64_t cubes, int64_t vectors,
                   Config &c, std::string &error) {
    if (cubes<=0 || vectors<cubes || vectors>255 || cubes>255) return Reject(error, "invalid core capacities");
    const auto cores = std::lldiv(vectors, cubes);
    if (cores.rem != 0) return Reject(error, "invalid core capacities");
    if (!ValidateAttrs(a, error) || !ReadQuery(x, c, error) || !ReadSequence(x, a, c, error) ||
        !ReadKey(x, a, c, error)) return false;
    c.max_q=a.max_q<=0 ? c.rows : a.max_q;
    const int64_t shard_tiles=Ceil(c.max_k,2048), actual_shard=shard_tiles*256;
    c.parts=Ceil(Ceil(c.max_k,256),shard_tiles);
    c.fast = UseFastSchedule(x, a, c, actual_shard);
    const int64_t groups=c.batches*Ceil(c.max_q,6);
    ChooseShards(c, cubes, groups, shard_tiles);
    ConfigureSelection(x, a, c, groups, actual_shard, cubes, cores.quot);
    return true;
}
}
} // namespace
