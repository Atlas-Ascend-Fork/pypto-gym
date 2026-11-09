# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-Pro quant_mla_prolog CPU golden reference (higher precision).

纯 torch CPU 真值。供 test_quant_mla_prolog.py 的精度校验使用。

与 NPU 实现的数值差异:
  - QA 下投影用 FP64 累加（FP8 操作数与 E8M0 descale 都是二进位精确值，
    CPU BLAS 的 FP32 累加会把结果推过 BF16/MXFP8 中点，改变量化档位）。
  - KVA 下投影保持 FP32（KV 路径下游无 BF16/MX 档位判定）。
  - 输出保留 FP32 中间量，同时给出编码后 FP8 版本。

精度细节（与 NPU golden 同源）:
  1. γ_qa / γ_kva 走 FP32→BF16→FP32 一圈
  2. RMSNorm 源序 FP32 平方和 + sqrt 后除，不用 rsqrt
  3. QA 重量化: FP32→BF16 用 ties-away(RNA)，BF16→E4M3 用 RNE
  4. Q 输出量化的除法用 nextafter-toward-zero 模拟 NPU FP32 除法
"""

__all__ = ["make_inputs", "quant_mla_prolog_golden_cpu"]

import torch

DIM, Q_LORA, KV_LORA = 7168, 1536, 512
N_HEADS, QK_NOPE, ROPE = 96, 128, 64
HEAD_DIM = QK_NOPE + ROPE          # 192
FEATURES = KV_LORA + ROPE          # 576
FP8_MAX = 448.0
PA_BLOCK_SIZE = 128
MX_BLOCK = 32

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


# ─────────────────────────────────────────────
# 位级辅助（与 NPU golden 逐函数一致）
# ─────────────────────────────────────────────

def _bf16_roundtrip(value: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16 -> FP32 一圈。精度细节 1（原版 golden.py:462-463）。"""
    return value.to(BF16).float()


def cast_bf16_round_away(value: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16，ties-away。精度细节 3 之 ✂①（原版 golden.py:219）。"""
    value = value.float().contiguous()
    bits = value.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (bits >> 16) + ((bits & 0xFFFF) >= 0x8000).to(torch.int64)
    return (upper & 0xFFFF).to(torch.int16).view(BF16)


def decode_e8m0(storage: torch.Tensor, outer: int, k: int) -> torch.Tensor:
    """E8M0 块 scale -> FP32 乘数 [outer, k]（原版 golden.py:209）。"""
    raw = storage.contiguous().view(torch.uint8)
    values = raw.reshape(outer, -1)[:, : (k + MX_BLOCK - 1) // MX_BLOCK].float() - 127
    return torch.pow(2.0, values).repeat_interleave(MX_BLOCK, dim=1)[:, :k]


def unpack_scale_pair_nz(storage: torch.Tensor, outer: int, groups: int) -> torch.Tensor:
    """NZ 物理 MX scale -> 逻辑 [outer, groups, 2]（原版 golden.py:173）。"""
    gp = storage.shape[1]
    data = storage.view(torch.uint8).permute(0, 2, 1, 3).contiguous().reshape(-1, gp, 2)
    return data[:outer, :groups].contiguous().view(E8M0)


def logical_descale(storage: torch.Tensor, outer: int, k: int) -> torch.Tensor:
    """ND(rank-3) / NZ(rank-4) MX scale 统一成逻辑形状（原版 golden.py:189）。"""
    groups = (k + 63) // 64
    if storage.ndim == 3:
        return storage
    if storage.ndim != 4:
        raise ValueError("MX scale must be logical rank-3 ND or physical rank-4 NZ")
    return unpack_scale_pair_nz(storage, outer, groups)


def unpack_nz(weight: torch.Tensor) -> torch.Tensor:
    """E4M3 权重 NZ [O/32,K,32] -> 逻辑 [O,K]（原版 golden.py:141）。"""
    return (
        weight.permute(0, 2, 1)
        .reshape(weight.shape[0] * 32, weight.shape[1])
        .contiguous()
    )


def dynamic_mx_quant_bf16(value: torch.Tensor):
    """BF16 block-32 MX 量化，含 E=0 处理。精度细节 3 之 ✂②（原版 golden.py:230）。"""
    t, k = value.shape
    exponent = (value.contiguous().view(torch.int16).to(torch.int32) & 0x7F80) >> 7
    maximum = exponent.reshape(t, k // MX_BLOCK, MX_BLOCK).amax(-1)
    raw = (maximum - 8).clamp_min(0)
    inverse_bits = (0x7F00 - (raw << 7)).to(torch.int16)
    inverse_bits = torch.where(
        maximum == 0, torch.zeros_like(inverse_bits), inverse_bits
    )
    inverse = inverse_bits.contiguous().view(BF16).repeat_interleave(MX_BLOCK, dim=1)
    normalized = (value.float() * inverse.float()).to(BF16).float()
    encoded = normalized.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    scale = torch.zeros((t, (k + 63) // 64 * 2), dtype=torch.uint8)
    scale[:, : k // MX_BLOCK] = raw.to(torch.uint8)
    return encoded, scale.reshape(t, (k + 63) // 64, 2).view(E8M0)


def rms_norm_fp32(value: torch.Tensor, weight: torch.Tensor, eps: float):
    """FP32 RMSNorm，源序平方和 + sqrt 后除。精度细节 2（原版 golden.py:336）。"""
    x = value.float()
    denominator = torch.sqrt(
        x.square().sum(-1, keepdim=True) * (1.0 / x.shape[-1]) + eps
    )
    return (x / denominator) * weight


def npu_divide_fp32_cpu(numerator: torch.Tensor, denominator: torch.Tensor):
    """在 CPU 上模拟 NPU 的 FP32 除法结果。精度细节 4（原版 golden.py:347）。

    原版注释: 在 E4M3 的精确中点上，设备商比 torch CPU 除法少一个 FP32 ULP
    （朝零方向）。那一个 ULP 会选到相邻的 FP8 档，所以在输出量化边界上必须保留。
    """
    quotient = numerator.float() / denominator.float()
    return torch.nextafter(quotient, torch.zeros_like(quotient))


def cache_c0(dtype) -> int:
    """PA_NZ 末维宽度：C1 的 FP8 是 32，C0 的 BF16 是 16。"""
    if dtype == BF16:
        return 16
    if dtype in (FP8, torch.uint8):
        return 32
    raise ValueError("Cache must be BF16 or 8-bit float storage")


def _raw(tensor: torch.Tensor) -> torch.Tensor:
    """按元素宽度取整数视图，使地址公式保持在元素单位。"""
    if tensor.dtype == BF16:
        return tensor.view(torch.int16)
    if tensor.dtype in (FP8, E8M0, torch.uint8):
        return tensor.view(torch.uint8)
    raise ValueError("Expected BF16 or an 8-bit float storage tensor")


def scatter_pa_nz(storage: torch.Tensor, indices: torch.Tensor, kv: torch.Tensor):
    """按 cache_index 散写 PA_NZ cache；slot < 0 整行跳过。

    地址公式照搬原版 golden.py:299 _cache_offsets()，单位是元素不是字节 ——
    C0 的 BF16 cache 走 int16 视图（36 块 × 16 列），C1 的 FP8 走 uint8（18 × 32）。
    """
    _blocks, f1, page, c0 = storage.shape
    if c0 != cache_c0(storage.dtype) or kv.dtype != storage.dtype:
        raise ValueError("PA_NZ storage c0/dtype mismatch")
    f = f1 * c0
    result = storage.clone()
    flat = _raw(result).reshape(-1)
    data = _raw(kv.contiguous())
    keep = indices >= 0
    if not bool(keep.any()):
        return result
    rows = torch.nonzero(keep, as_tuple=True)[0]
    slots = indices[keep].unsqueeze(1)
    d = torch.arange(f, dtype=torch.int64).unsqueeze(0)
    offsets = (
        (slots // page) * f * page
        + (d // c0) * page * c0
        + (slots % page) * c0
        + d % c0
    )
    flat[offsets.reshape(-1)] = data[rows].reshape(-1)
    return result


# ─────────────────────────────────────────────
# CPU Golden 参考实现（更高精度）
# ─────────────────────────────────────────────

def _check_golden_args(tensors, qscale_kv, norm_eps, quant_mode_aw,
                       quant_mode_c):
    """模式组合与 CPU 驻留校验。

    raise 的先后就是用例依赖的报错优先级：四条模式判据全部先于 CPU 驻留判据。
    """
    if quant_mode_aw != 1:
        raise ValueError("quant_mla_prolog_golden_cpu requires quant_mode_aw=1")
    if quant_mode_c not in (0, 1):
        raise ValueError("quant_mode_c must be 0 or 1")
    # C1 必须给 qscale_kv，C0 必须不给。
    if quant_mode_c == 1 and qscale_kv is None:
        raise ValueError("quant_mode_c=1 requires qscale_kv")
    if quant_mode_c == 0 and qscale_kv is not None:
        raise ValueError("quant_mode_c=0 must not pass qscale_kv")
    if not (isinstance(norm_eps, (int, float)) and norm_eps > 0):
        raise ValueError("norm_eps must be positive and finite")
    if any(t.device.type != "cpu" for t in tensors if t is not None):
        raise ValueError("quant_mla_prolog_golden_cpu requires CPU tensors")


def _dequant_inputs(x, wqa, wqb, wkva, descale_x, descale_wqa, descale_wqb,
                    descale_wkva, tokens):
    """激活与三个 NZ 权重的 MX 反量化（原版 golden.py:452-458）。"""
    x_f = x.float() * decode_e8m0(
        logical_descale(descale_x, tokens, DIM), tokens, DIM
    )
    wqa_f = unpack_nz(wqa).float()
    wqa_f = wqa_f * decode_e8m0(
        logical_descale(descale_wqa, Q_LORA, DIM), Q_LORA, DIM
    )
    wqb_f = unpack_nz(wqb).float()
    wqb_f = wqb_f * decode_e8m0(
        logical_descale(descale_wqb, N_HEADS * HEAD_DIM, Q_LORA),
        N_HEADS * HEAD_DIM,
        Q_LORA,
    )
    wkva_f = unpack_nz(wkva).float()
    wkva_f = wkva_f * decode_e8m0(
        logical_descale(descale_wkva, FEATURES, DIM), FEATURES, DIM
    )
    return x_f, wqa_f, wqb_f, wkva_f



def quant_mla_prolog_golden_cpu(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    descale_x: torch.Tensor,
    descale_wqa: torch.Tensor,
    descale_wqb: torch.Tensor,
    descale_wkva: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qscale_kv: torch.Tensor,
    norm_eps: float = 1e-06,
    quant_mode_aw: int = 1,
    quant_mode_c: int = 1,
):
    """CPU 更高精度 golden。参数含义同 quant_mla_prolog_golden。

    与 NPU golden 数学逻辑一致，但:
      - QA 下投影 matmul 用 FP64 累加，KVA 下投影保持 FP32（原版 golden.py:468/482 的理由）
      - 返回量化前的 FP32 中间量，不只给 FP8 编码

    Returns:
        dict:
          q            [T,96,576] FP8_E4M3   编码后的 Q
          descale_q    [T,96]     FP32       per-token-head scale
          kv_cache_out [blk,18,128,32] FP8   散写后的 cache
          q_fp32       [T,96,576] FP32       量化前的 Q（更高精度参考）
          kv_fp32      [T,576]    FP32       量化前的 KV 行（更高精度参考）
          qa_mx        [T,1536]   FP8_E4M3   中间量，供逐相位定位
          qa_scale     [T,24,2]   E8M0       中间量
    """
    _check_golden_args(
        (x, wqa, wqb, wkva, wkb, descale_x, descale_wqa, descale_wqb,
         descale_wkva, norm_weight_qa, norm_weight_kva, kv_cache,
         cache_index, qscale_kv),
        qscale_kv, norm_eps, quant_mode_aw, quant_mode_c)

    tokens = x.shape[0]

    x_f, wqa_f, wqb_f, wkva_f = _dequant_inputs(
        x, wqa, wqb, wkva, descale_x, descale_wqa, descale_wqb,
        descale_wkva, tokens)

    # ---- γ 的 BF16 一圈（精度细节 1）----
    gamma_qa = _bf16_roundtrip(norm_weight_qa)
    gamma_kva = _bf16_roundtrip(norm_weight_kva)

    # ---- Phase QA（FP64 累加，原版 golden.py:464-468）----
    qa_raw = (x_f.double() @ wqa_f.double().t()).float()

    # ---- Phase Q: RMSNorm + 重量化 ----
    qa_fp32 = rms_norm_fp32(qa_raw, gamma_qa, norm_eps)
    qa_bf16 = cast_bf16_round_away(qa_fp32)                  # ✂① ties-away
    qa_mx, qa_scale = dynamic_mx_quant_bf16(qa_bf16)         # ✂② RNE -> E4M3
    qa_deq = qa_mx.float() * decode_e8m0(qa_scale, tokens, Q_LORA)

    # ---- Phase H1-QB ----
    # 原版 golden.py:473 这一处用 FP32（qa_deq 与 wqb_f 都已是反量化后的 FP32）
    q_states = (qa_deq @ wqb_f.t()).reshape(tokens, N_HEADS, HEAD_DIM)
    q_pass = q_states[..., :QK_NOPE].to(BF16)                # ✂③
    q_rot = q_states[..., QK_NOPE:].to(BF16)                 # ✂③

    # ---- Phase H1-WKB ----
    # FP32 累加后降 BF16，对齐 kernel 的 L0C FP32 + ✂⑤（SPEC B2）。
    # 原版 golden.py:479 用 BF16 einsum；此处与 NPU golden 保持一致地走 FP32 累加。
    q_latent = torch.einsum("tnd,ndk->tnk", q_pass.float(), wkb.float()).to(BF16)

    # ---- Phase H1-Q epilogue ----
    q_combined = torch.cat((q_latent, q_rot), -1)
    q_fp32 = q_combined.float()
    if quant_mode_c == 1:
        descale_q = q_fp32.abs().amax(-1).clamp_min(1e-8) / FP8_MAX
        # 精度细节 4: 这一处除法用 NPU 语义（nextafter toward zero）
        q_normalized = npu_divide_fp32_cpu(q_fp32, descale_q.unsqueeze(-1))
        q_out = q_normalized.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    else:
        # C0 无输出量化：q_combined 已是 BF16（q_latent / q_rot 两段都降过），
        # 直接当终输出，没有 descale_q。
        descale_q = None
        q_out = q_combined

    # ---- Phase KVA + KV epilogue ----
    # 原版 golden.py:482 这一处是 FP32，不升 FP64（见文件头说明）
    kv_raw = x_f @ wkva_f.t()
    kv_norm = rms_norm_fp32(kv_raw[..., :KV_LORA], gamma_kva, norm_eps)
    k_rot = _bf16_roundtrip(kv_raw[..., KV_LORA:])
    kv_fp32 = torch.cat((kv_norm, k_rot), -1)
    if quant_mode_c == 1:
        # KV cache 这一路用普通 CPU 除法语义（原版 golden.py:501-505 明确说
        # 上面那个 one-ULP 模型只针对 split-Q 的输出量化边界）
        kv_store = (kv_fp32 / qscale_kv).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    else:
        # C0：norm 后直接降 BF16，RoPE 段本就是 BF16 值
        kv_store = kv_fp32.to(BF16)
    kv_cache_out = scatter_pa_nz(kv_cache, cache_index, kv_store)

    outputs = {
        "q": q_out,
        "kv_cache_out": kv_cache_out,
        "q_fp32": q_fp32,
        "kv_fp32": kv_fp32,
        "qa_mx": qa_mx,
        "qa_scale": qa_scale,
    }
    if descale_q is not None:
        outputs["descale_q"] = descale_q
    return outputs


# ==========================================
# 输入构造
# ==========================================

def make_inputs(tokens=2, blocks=2, seed=20260916, quant_mode_c=1):
    """自验用的小规模输入（模型维度仍是固定的 7168/1536/512/96）。

    quant_mode_c=0 时 cache 换成 BF16 的 36 块 × 16 列、qscale_kv 给 None
    （C0 传了要报错，见 _validate_quant_mla_modes 的两条判据）。

    Stage 4 精度对比应复用 NPU golden 的 _make_inputs(device=cpu)，
    那边是 SPEC §12 的 P0 配置；这里刻意缩小 token 数与 block 数，
    让 _validate() 在 CPU 上尽快跑完（P0 的 7168x18432 投影在 CPU 上很慢）。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cache_dtype = FP8 if quant_mode_c else BF16
    cc = cache_c0(cache_dtype)

    def _descale(outer, k):
        groups = (k + 63) // 64
        raw = (
            125
            + (
                3 * torch.arange(outer)[:, None, None]
                + 2 * torch.arange(groups)[None, :, None]
                + torch.arange(2)[None, None, :]
            )
            % 5
        ).to(torch.uint8)
        return raw.contiguous().view(E8M0)

    def _weight_nz(out_channels, k):
        logical = (torch.randn(out_channels, k, generator=gen) * 0.5).to(FP8)
        return logical.reshape(out_channels // 32, 32, k).permute(0, 2, 1).contiguous()

    x = (torch.randn(tokens, DIM, generator=gen) * 0.5).to(FP8)
    args = [
        x,
        _weight_nz(Q_LORA, DIM),
        _weight_nz(N_HEADS * HEAD_DIM, Q_LORA),
        _weight_nz(FEATURES, DIM),
        torch.randn(N_HEADS, QK_NOPE, KV_LORA, generator=gen).to(BF16),
        _descale(tokens, DIM),
        _descale(Q_LORA, DIM),
        _descale(N_HEADS * HEAD_DIM, Q_LORA),
        _descale(FEATURES, DIM),
        torch.randn(Q_LORA, generator=gen),
        torch.randn(KV_LORA, generator=gen),
        torch.zeros((blocks, FEATURES // cc, PA_BLOCK_SIZE, cc),
                    dtype=cache_dtype),
        # slot 互异且落在 blocks*128 容量内（跨页分布，覆盖 page/row 两段地址）
        torch.tensor(
            [(i * 127 + 3) % (blocks * PA_BLOCK_SIZE) for i in range(tokens)],
            dtype=torch.int64,
        ),
        torch.tensor([0.5], dtype=torch.float32) if quant_mode_c else None,
    ]
    return args, {"norm_eps": 1e-06, "quant_mode_aw": 1,
                  "quant_mode_c": quant_mode_c}


def _validate_c0():
    """C0 分支自验：输出规格、PA_NZ 36x16 地址公式回环、入参校验两条。

    散写回环是这里唯一不可替代的一项：C0 把 cache 的物理分块从 18 块 x 32 列
    换成 36 块 x 16 列，地址公式写错不一定在 L2 上露出来，所以直接按
    cache_index 取回写入行、与 kv_store 逐 bit 比。
    """
    args, kwargs = make_inputs(tokens=4, blocks=2, quant_mode_c=0)
    out = quant_mla_prolog_golden_cpu(*args, **kwargs)
    tokens = args[0].shape[0]
    cache = args[11]
    page = cache.shape[2]

    checks = [
        ("q dtype = BF16", out["q"].dtype == BF16),
        ("q shape", tuple(out["q"].shape) == (tokens, N_HEADS, FEATURES)),
        ("没有 descale_q 这一路", "descale_q" not in out),
        ("kv_cache_out dtype = BF16", out["kv_cache_out"].dtype == BF16),
        ("kv_cache_out 分块 = 36 x 16",
         tuple(cache.shape[1:]) == (FEATURES // 16, page, 16)),
        ("q / kv_cache_out 全部有限",
         bool(torch.isfinite(out["q"].float()).all()
              and torch.isfinite(out["kv_cache_out"].float()).all())),
    ]

    # 散写回环：取回写入行，与 kv_fp32 降 BF16 后逐 bit 比
    want = out["kv_fp32"].to(BF16)
    got = out["kv_cache_out"]
    rows_ok = True
    for t in range(tokens):
        slot = int(args[12][t])
        if slot < 0:
            continue
        row = got[slot // page, :, slot % page, :].reshape(-1)
        rows_ok &= bool(torch.equal(row.view(torch.int16),
                                    want[t].view(torch.int16)))
    checks.append(("散写回环逐 bit 一致（36 块 x 16 列地址公式）", rows_ok))

    # 入参校验两条
    def _raises(mutate):
        bad = list(args)
        bad_kw = dict(kwargs)
        mutate(bad, bad_kw)
        try:
            quant_mla_prolog_golden_cpu(*bad, **bad_kw)
        except ValueError:
            return True
        return False

    def _c0_with_scale(a, kw):
        a[13] = torch.tensor([0.5], dtype=torch.float32)

    def _c1_without_scale(a, kw):
        kw["quant_mode_c"] = 1

    checks.append(("C0 传了 qscale_kv 要报错", _raises(_c0_with_scale)))
    checks.append(("C1 缺 qscale_kv 要报错", _raises(_c1_without_scale)))

    failed = [label for label, ok in checks if not ok]
    if failed:
        raise AssertionError("C0 分支自检失败项：" + "；".join(failed))


def _validate():
    """验证 CPU golden 可运行、输出 dtype 与值域正确。

    通过时无输出、返回即成立；任一项不成立就抛 AssertionError，消息里带上
    失败项的名字，脚本跑法由此得到非零退出码。
    """
    args, kwargs = make_inputs(tokens=2, blocks=2)
    out = quant_mla_prolog_golden_cpu(*args, **kwargs)
    tokens = args[0].shape[0]

    checks = [
        ("q shape", tuple(out["q"].shape) == (tokens, N_HEADS, FEATURES)),
        ("q dtype = E4M3", out["q"].dtype == FP8),
        ("descale_q shape", tuple(out["descale_q"].shape) == (tokens, N_HEADS)),
        ("descale_q dtype = FP32", out["descale_q"].dtype == torch.float32),
        ("kv_cache_out shape", out["kv_cache_out"].shape == args[11].shape),
        ("kv_cache_out dtype = E4M3", out["kv_cache_out"].dtype == FP8),
        # 更高精度参考量必须是 FP32，不降回原 dtype
        ("q_fp32 dtype = FP32", out["q_fp32"].dtype == torch.float32),
        ("q_fp32 shape", tuple(out["q_fp32"].shape) == (tokens, N_HEADS, FEATURES)),
        ("kv_fp32 dtype = FP32", out["kv_fp32"].dtype == torch.float32),
        ("kv_fp32 shape", tuple(out["kv_fp32"].shape) == (tokens, FEATURES)),
    ]
    failed = [label for label, ok in checks if not ok]
    if failed:
        raise AssertionError("输出规格检查失败项：" + "；".join(failed))

    nonfinite = []
    for name in ("q", "descale_q", "kv_cache_out", "q_fp32", "kv_fp32"):
        if not bool(torch.isfinite(out[name].float()).all()):
            nonfinite.append(name)
    if nonfinite:
        raise AssertionError(
            "以下输出含非有限值：" + "；".join(nonfinite))

    range_checks = [
        ("|q| <= 448", bool((out["q"].float().abs() <= FP8_MAX).all())),
        ("|kv_cache_out| <= 448",
         bool((out["kv_cache_out"].float().abs() <= FP8_MAX).all())),
        ("descale_q > 0", bool((out["descale_q"] > 0).all())),
    ]
    failed = [label for label, ok in range_checks if not ok]
    if failed:
        raise AssertionError("值域检查失败项：" + "；".join(failed))

    neg_args = list(args)
    neg_index = args[12].clone()
    neg_index[0] = -1
    neg_args[12] = neg_index
    neg = quant_mla_prolog_golden_cpu(*neg_args, **kwargs)
    page = args[11].shape[2]
    slot = int(args[12][0])
    base = args[11].view(torch.uint8)
    got = neg["kv_cache_out"].view(torch.uint8)
    if not torch.equal(got[slot // page, :, slot % page, :],
                       base[slot // page, :, slot % page, :]):
        raise AssertionError(
            f"边界 case 失败：cache_index[0]=-1 时 slot {slot} 仍被写")

    _validate_c0()
