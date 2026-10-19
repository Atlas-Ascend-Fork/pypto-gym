#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
'''
'''
import os
import math
import logging
from pathlib import Path

import torch
import torch_npu
import pytest

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import numpy as np
import pypto

from deepseek_v4.mla_prolog_quant_v4_impl import mla_prolog_v4, MlaPrologV4Attrs, \
    MlaTileConfigs, MlaPrologV4Configs, check_input_output_shape_dtype, mla_prolog_quant_pypto, \
    MlaCheckInputConfig
from tests.ops.utils.compare import compare

torch.manual_seed(5)


def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True


def rms_norm_new(x, eps=1e-6):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff
    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res = x_f32 / reduce_sqrt
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rms_norm(x, gamma, eps=1e-6):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    gamma = gamma.to(torch.float32)
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff
    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=1):
    input_dtype = q.dtype
    q_clone = q.clone()
    k_clone = k.clone()
    t, nq, d = q.shape
    t, nk, d = k.shape
    q = q.reshape(t, nq, d//2, 2).permute(0, 1, 3, 2).reshape(t, nq, d)
    k = k.reshape(t, nk, d//2, 2).permute(0, 1, 3, 2).reshape(t, nk, d)
    q_t = rotate_half(q)
    k_t = rotate_half(k)
    q_new = q_t.reshape(t, nq, 2, d//2).permute(0, 1, 3, 2).reshape(t, nq, d)
    k_new = k_t.reshape(t, nk, 2, d//2).permute(0, 1, 3, 2).reshape(t, nk, d)
    q_new = q_new.to(torch.float32)
    k_new = k_new.to(torch.float32)
    cos = torch.unsqueeze(cos, dim=unsqueeze_dim)
    sin = torch.unsqueeze(sin, dim=unsqueeze_dim)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    q_embed = q_clone * cos + q_new * sin
    k_embed = k_clone * cos + k_new * sin
    if input_dtype != torch.float32:
        q_embed, k_embed = q_embed.to(input_dtype), k_embed.to(input_dtype)
    return q_embed, k_embed


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).cpu().numpy().tobytes())


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.to(torch.float32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = torch.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
    max_value = torch.max(abs_res, dim=reduce_idx, keepdims=True)[0]
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_fp16 = out_int32.to(torch.float16)
    out_int8 = torch.trunc(out_fp16).to(torch.int8)
    scale_dequant = 1 / scale_quant
    return out_int8, scale_dequant


def mla_prolog_compute(inputs):
    dtype = inputs.get("dtype")
    x = inputs.get("x")
    wq_a = inputs.get("wq_a")
    wq_b = inputs.get("wq_b")
    qk_rope_head_dim = inputs.get("qk_rope_head_dim")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    w_kv = inputs.get("w_kv")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    wq_b_scale = inputs.get("wq_b_scale")

    t, h = x.shape
    qk_rope_head_dim = cos.shape[1]
    head_dim = w_kv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim

    ''' q '''
    q_a_proj = torch.matmul(x.to(torch.float32), wq_a.to(torch.float32))
    q_a_layernorm = rms_norm(q_a_proj, gamma_cq)
    q_a_layernorm_out = q_a_layernorm.to(torch.bfloat16)
    q_a_quant, q_a_quant_scale = quant(q_a_layernorm, True)
    q_b_proj = torch.matmul(q_a_quant.to(torch.int32), wq_b.to(torch.int32))
    ''' dequant'''
    q_b_proj_fp32 = q_b_proj.to(torch.float32)
    q_b_proj_dequant = q_b_proj_fp32 * q_a_quant_scale
    q_b_proj_dequant = q_b_proj_dequant * wq_b_scale

    q_reshape = q_b_proj_dequant.reshape(t, num_heads, head_dim)
    q_reshape = rms_norm_new(q_reshape)
    q_reshape = q_reshape.to(torch.bfloat16)

    """ kv """
    kv_a_proj = torch.matmul(x.to(torch.float32), w_kv.to(torch.float32))
    kv_a_proj_norm = rms_norm(kv_a_proj, gamma_ckv)
    kv_reshape = kv_a_proj_norm.reshape(t, head_dim)
    kv_reshape = kv_reshape.to(torch.bfloat16)
    """ rope"""
    q_pe = q_reshape[:, :, -qk_rope_head_dim:]
    k_pe = kv_reshape[:, -qk_rope_head_dim:]
    k_pe_r = k_pe.reshape(t, 1, qk_rope_head_dim)
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 1)
    k_embed_r = k_embed.reshape(t, qk_rope_head_dim)
    q_out = torch.concat([q_reshape[:, :, :-qk_rope_head_dim], q_embed], -1)
    kv_out = torch.concat([kv_reshape[:, :-qk_rope_head_dim], k_embed_r], -1)
    return q_out, kv_out, q_a_quant, q_a_quant_scale


def gen_mla_prolog_input_data(params, dtypes, is_quant=True, is_nz=False):
    dtype = dtypes
    t = params.get("t")
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    head_dim = params.get("head_dim")
    rope_head_dim = params.get("qk_rope_head_dim")

    x_shape = [t, h]
    wq_a_shape = [h, q_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [head_dim]
    wq_b_shape = [q_lora_rank, n * head_dim]
    wkv_shape = [h, head_dim]
    cos_shape = [t, rope_head_dim]

    res = [None] * 9
    x = torch.empty(x_shape, dtype=dtype).uniform_(-1, 1)
    res[0] = x
    wq_a = torch.empty(wq_a_shape, dtype=dtype).uniform_(-0.1, 0.1)
    wq_b = torch.empty(wq_b_shape, dtype=dtype).uniform_(-0.1, 0.1)
    w_kv = torch.empty(wkv_shape, dtype=dtype).uniform_(-0.1, 0.1)
    if is_quant:
        wq_b, wq_b_scale = quant(wq_b, False)
        res[-1] = wq_b_scale
    res[1] = wq_a
    res[2] = wq_b
    res[3] = w_kv
    gamma_cq = torch.empty(gamma_cq_shape, dtype=dtype).uniform_(-1, 1)
    gamma_ckv = torch.empty(gamma_ckv_shape, dtype=dtype).uniform_(-1, 1)
    res[4] = gamma_cq
    res[5] = gamma_ckv
    cos = torch.empty(cos_shape, dtype=dtype).uniform_(-1, 1)
    sin = torch.empty(cos_shape, dtype=dtype).uniform_(-1, 1)
    res[6] = cos
    res[7] = sin
    return res


def gen_mla_prolog_data(params, dtype, is_quant=True, is_nz=False):
    x, wq_a, wq_b, w_kv, gamma_cq, gamma_ckv, cos, sin, wq_b_scale = \
        gen_mla_prolog_input_data(params, dtype, is_quant, is_nz)
    inputs = dict()
    inputs["x"] = x
    inputs["wq_a"] = wq_a
    inputs["wq_b"] = wq_b
    inputs["w_kv"] = w_kv
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["wq_b_scale"] = wq_b_scale
    q_out, kv_out, qr_out, qr_scale_out = mla_prolog_compute(inputs)
    outputs = {"q_golden": q_out, "kv_golden": kv_out,
               "qr_golden": qr_out, "qr_scale_golden": qr_scale_out}
    return inputs, outputs


def convert_pypto_to_torch_type(pypto_type):
    if pypto_type == pypto.DataType.DT_INT8:
        return torch.int8
    elif pypto_type == pypto.DataType.DT_INT32:
        return torch.int32
    elif pypto_type == pypto.DataType.DT_FP32:
        return torch.float32
    elif pypto_type == pypto.DataType.DT_FP16:
        return torch.float16
    elif pypto_type == pypto.DataType.DT_BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported pypto.DataType: {pypto_type}")


def _prepare_mla_tensors(params, input_tensors, is_nz):
    """Reshape + move input tensors to NPU, optionally apply FRACTAL_NZ."""
    t = params['t']
    n1 = params["num_heads"]
    h = params["h"]
    q_lora_rank = params["q_lora_rank"]
    qk_rope_head_dim = params["qk_rope_head_dim"]
    head_dim = params["head_dim"]

    wq_a_shape = [h, q_lora_rank]
    wq_b_shape = [q_lora_rank, n1 * head_dim]
    wkv_shape = [h, head_dim]

    if is_nz:
        wq_a_nz = torch_npu.npu_format_cast(
            input_tensors["wq_a"].reshape(wq_a_shape).npu().contiguous(),
            torch_npu.Format.FRACTAL_NZ)
        wq_b_nz = torch_npu.npu_format_cast(
            input_tensors["wq_b"].reshape(wq_b_shape).npu().contiguous(),
            torch_npu.Format.FRACTAL_NZ)
        wkv_nz = torch_npu.npu_format_cast(
            input_tensors["w_kv"].reshape(wkv_shape).npu().contiguous(),
            torch_npu.Format.FRACTAL_NZ)
        input_tensors["wq_a"] = wq_a_nz
        input_tensors["wq_b"] = wq_b_nz
        input_tensors["w_kv"] = wkv_nz

    token_x = input_tensors["x"].reshape([t, h]).npu()
    wq_a = input_tensors["wq_a"].reshape(wq_a_shape).npu()
    wq_b = input_tensors["wq_b"].reshape(wq_b_shape).npu()
    wkv = input_tensors["w_kv"].reshape(wkv_shape).npu()
    rope_cos = input_tensors["cos"].reshape([t, qk_rope_head_dim]).npu()
    rope_sin = input_tensors["sin"].reshape([t, qk_rope_head_dim]).npu()
    gamma_cq = input_tensors["gamma_cq"].reshape([q_lora_rank]).npu()
    gamma_ckv = input_tensors["gamma_ckv"].reshape([head_dim]).npu()
    wq_b_scale = input_tensors["wq_b_scale"].reshape([n1 * head_dim, 1]).npu()
    return [token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv, wq_b_scale]


def _mla_prolog_output_shapes(params):
    """Return output tensor shapes for mla_prolog."""
    return ([params['t'], params["num_heads"], params["head_dim"]],
            [params['t'], params["head_dim"]],
            [params['t'], params["q_lora_rank"]],
            [params['t'], 1])


def _compare_mla_outputs(output_q_data, output_kv_data, output_qr_data, output_qr_scale_data,
                         golden_tensors, q_out_shape, kv_out_shape, qr_out_shape, qr_scale_shape):
    """Compare mla_prolog kernel outputs against golden tensors."""
    golden1 = golden_tensors["q_golden"].reshape(q_out_shape)
    golden2 = golden_tensors["kv_golden"].reshape(kv_out_shape)
    golden3 = golden_tensors["qr_golden"].reshape(qr_out_shape)
    golden4 = golden_tensors["qr_scale_golden"].reshape(qr_scale_shape)
    print("q ================")
    compare(output_q_data.cpu(), golden1.cpu(), "qOut", 0.0001, 0.0078125, 0.005)
    print("kv ================")
    compare(output_kv_data.cpu(), golden2.cpu(), "kvOut", 0.0001, 0.0078125, 0.005)
    print("qr ================")
    compare(output_qr_data.cpu(), golden3.cpu(), "qrOut", 1, 0, 0)
    print(" qr_scale ==========")
    compare(output_qr_scale_data.cpu(), golden4.cpu(), "qrScaleOut", 0.000025, 0.005, 0.005)
    print("=========== pass ==========")


class MLA_MODEL(torch.nn.Module):
    def forward(self, token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv, wq_b_scale):
        return mla_prolog_quant_pypto(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin,
                                      gamma_cq, gamma_ckv, wq_b_scale)


def mla_prolog(params, input_tensors, golden_tensors, dtype, is_nz):
    d_type = pypto.DataType.DT_FP16 if dtype == pypto.DataType.DT_FP16 else pypto.DataType.DT_BF16
    q_out_shape, kv_out_shape, qr_out_shape, qr_scale_shape = _mla_prolog_output_shapes(params)

    inputs = _prepare_mla_tensors(params, input_tensors, is_nz)

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    if pypto.platform.npuarch == 'DAV_3510':
        compiler_config.debug.aclgraph.clone_input = False
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(MLA_MODEL(), dynamic=False, fullgraph=True, backend=npu_backend)

    output_q_data, output_kv_data, output_qr_data, output_qr_scale_data = model(*inputs)
    pypto.runtime._device_synchronize()

    _compare_mla_outputs(output_q_data, output_kv_data, output_qr_data, output_qr_scale_data,
                         golden_tensors, q_out_shape, kv_out_shape, qr_out_shape, qr_scale_shape)


def mla_prolog_eager(params, input_tensors, golden_tensors, dtype, is_nz, attrs, configs):
    d_type = pypto.DataType.DT_FP16 if dtype == pypto.DataType.DT_FP16 else pypto.DataType.DT_BF16
    q_out_shape, kv_out_shape, qr_out_shape, qr_scale_shape = _mla_prolog_output_shapes(params)

    inputs = _prepare_mla_tensors(params, input_tensors, is_nz)
    token_x = inputs[0]

    output_q_data = torch.empty([token_x.size(0), inputs[2].size(1) // inputs[7].size(0),
                                 inputs[7].size(0)], dtype=token_x.dtype, device=f'{token_x.device}')
    output_kv_data = torch.empty([token_x.size(0), inputs[7].size(0)],
                                 dtype=token_x.dtype, device=f'{token_x.device}')
    output_qr_data = torch.empty([token_x.size(0), inputs[1].size(1)],
                                 dtype=torch.int8, device=f'{token_x.device}')
    output_qr_scale_data = torch.empty([token_x.size(0), 1],
                                       dtype=torch.float32, device=f'{token_x.device}')
    check_config = MlaCheckInputConfig(
        token_x=token_x, wq_a=inputs[1], wq_b=inputs[2], wkv=inputs[3],
        rope_cos=inputs[4], rope_sin=inputs[5], gamma_cq=inputs[6],
        gamma_ckv=inputs[7], wq_b_scale=inputs[8],
        output_q_data=output_q_data, output_kv_data=output_kv_data,
        output_qr_data=output_qr_data, output_qr_scale_data=output_qr_scale_data)
    check_input_output_shape_dtype(check_config)

    tile_configs = MlaTileConfigs(
        two_dim_tile=[1, 64], three_dim_tile=[1, 64, 64],
        four_dim_tile=[1, 64, 64, 64],
        vec_tile=[max(1, token_x.shape[0] // 16), 64])

    params_info = [
        token_x, inputs[1], inputs[2], inputs[3], inputs[6],
        inputs[7], inputs[4], inputs[5], inputs[8], 
        output_q_data, output_kv_data, output_qr_data, output_qr_scale_data]
    mla_prolog_v4(*params_info, attrs, configs, tile_configs)
    pypto.runtime._device_synchronize()

    _compare_mla_outputs(output_q_data, output_kv_data, output_qr_data, output_qr_scale_data,
                         golden_tensors, q_out_shape, kv_out_shape, qr_out_shape, qr_scale_shape)


def _mla_prolog_test_params(t=4, is_quant=True, is_nz=True):
    """Return standard test params, dtype, and tensor gen flags."""
    params = {
        't': t, 'num_heads': 64, 'h': 4096,
        'q_lora_rank': 1024, 'head_dim': 512, 'qk_rope_head_dim': 64,
    }
    dtype = pypto.DataType.DT_BF16
    return params, dtype, is_quant, is_nz


@pytest.mark.skip("t=4")
def test_t4_pa_nd_bf16():
    prep_env()
    params, dtype, is_quant, is_nz = _mla_prolog_test_params(t=4)
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_quant, is_nz)
    mla_prolog(params, input_tensors, golden_data, dtype, is_nz)


@pytest.mark.skip("t=16")
def test_t4_pa_nd_bf16_eager():
    prep_env()
    params, dtype, is_quant, is_nz = _mla_prolog_test_params(t=4)
    attrs = MlaPrologV4Attrs(eps=1e-6)
    configs = MlaPrologV4Configs(
        unroll_list=[128, 64, 32, 16, 1], cube_l1_reuse_setting={2: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024, pg_upper_bound=8192,
        block_size=128, t_sub_tile=1, chunk_size=2)
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_quant, is_nz)
    mla_prolog_eager(params, input_tensors, golden_data, dtype, is_nz, attrs, configs)


@pytest.mark.skip("t=16")
def test_t16_pa_nd_bf16():
    prep_env()
    params, dtype, is_quant, is_nz = _mla_prolog_test_params(t=16)
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_quant, is_nz)
    mla_prolog(params, input_tensors, golden_data, dtype, is_nz)


@pytest.mark.skip("t=16")
def test_t16_pa_nd_bf16_eager():
    prep_env()
    params, dtype, is_quant, is_nz = _mla_prolog_test_params(t=16)
    attrs = MlaPrologV4Attrs(eps=1e-6)
    configs = MlaPrologV4Configs(
        unroll_list=[128, 64, 32, 16, 1], cube_l1_reuse_setting={2: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024, pg_upper_bound=8192,
        block_size=128, t_sub_tile=1, chunk_size=2)
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_quant, is_nz)
    mla_prolog_eager(params, input_tensors, golden_data, dtype, is_nz, attrs, configs)


@pytest.mark.skip("t=512")
def test_t512_pa_nd_bf16():
    prep_env()
    params, dtype, is_quant, is_nz = _mla_prolog_test_params(t=512)
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_quant, is_nz)
    mla_prolog(params, input_tensors, golden_data, dtype, is_nz)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO)
    test_t4_pa_nd_bf16()
