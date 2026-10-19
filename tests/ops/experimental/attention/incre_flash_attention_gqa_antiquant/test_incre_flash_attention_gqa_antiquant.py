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

"""Incremental Flash Attention with Grouped Query Attention (GQA) and Anti-Quantization.

This module implements incremental flash attention mechanism with GQA support
and anti-quantization for KV cache optimization.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import sys
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest


_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))
from experimental.attention.incre_flash_attention_gqa_antiquant.utils import create_logger, get_device, compare
from experimental.attention.incre_flash_attention_gqa_antiquant.incre_flash_attention_gqa_antiquant_impl import (
    incre_flash_attention_gqa_antiquant, AttentionTileConfig
)


logger = create_logger(__name__)


@dataclass
class IfaGqaConfig:
    """Configuration parameters for IFA GQA computation.

    Attributes:
        layout: Input layout format.
        b: Batch size.
        n1: Number of query heads.
        s1: Query sequence length.
        q_d: Query head dimension.
        n2: Number of key/value heads (for grouped query attention).
        s2: Key/Value sequence length (maximum).
        kv_d: Key/Value head dimension.
        block_size: Size of each block in paged KV cache.
        softmax_scale: Scaling factor for softmax.
    """

    layout: str = "BNSD"
    b: int = 72
    n1: int = 8
    s1: int = 1
    q_d: int = 128
    n2: int = 1
    s2: int | list = 2048
    kv_d: int = 128
    block_size: int = 128
    softmax_scale: float = 128 ** -0.5


def kv_cache_concat(cache_tensor: torch.Tensor,
                    kv_actual_seqs: torch.Tensor,
                    block_table: torch.Tensor,
                    ifa_gqa_config: IfaGqaConfig,
                    max_s2: int,
                    device: str) -> torch.Tensor:
    """Concatenate KV cache blocks into continuous tensors.

    Processed on CPU to support float8_e4m3fn dtype conversion,
    then moved to target device as bfloat16.

    Args:
        cache_tensor: KV cache tensor in block format (float8_e4m3fn).
        kv_actual_seqs: Actual sequence lengths for each batch.
        block_table: Mapping from logical to physical block indices.
        ifa_gqa_config: IFA GQA configuration.
        max_s2: max kv actual sequence length
        device: Device to create tensors on.

    Returns:
        Concatenated KV tensor in BSND format (bfloat16).
    """
    batch_size = ifa_gqa_config.b
    kv_num_heads = ifa_gqa_config.n2
    d = cache_tensor.shape[3]

    block_size = ifa_gqa_config.block_size
    dtype = cache_tensor.dtype

    cache_tensor_cpu = cache_tensor.cpu()

    result = torch.zeros([batch_size, kv_num_heads, max_s2, d], dtype=torch.bfloat16)

    for b_idx in range(batch_size):
        cur_kv_seq_len = kv_actual_seqs[b_idx].item()
        block_list = block_table[b_idx].cpu()
        temp_tensor = torch.zeros([1, kv_num_heads, cur_kv_seq_len, d], dtype=torch.bfloat16, device="cpu")
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = min((s_idx + 1) * block_size, cur_kv_seq_len)
            temp_tensor[:, :, start_idx:end_idx, :] = cache_tensor_cpu[
                block_idx:block_idx + 1, :, :end_idx - start_idx, :
            ].to(torch.bfloat16)
            s_idx += 1

        result[b_idx:b_idx + 1, :, :cur_kv_seq_len, :] = temp_tensor

    return result.to(device=device)


def gen_block_table(ifa_gqa_config: IfaGqaConfig,
                    kv_actual_seqs: torch.Tensor,
                    max_s2: int,
                    s2_tile: int,
                    device: str) -> torch.Tensor:
    """Generate a block table for paged KV cache.

    Args:
        ifa_gqa_config: IFA GQA configuration.
        kv_actual_seqs: Actual sequence lengths for each batch.
        s2_tile: s2_tile
        max_s2: max kv actual sequence length
        device: Device to create tensors on.

    Returns:
        Block table tensor of shape [batch_size, max_blocks_per_query].
    """
    block_num_per_batch = []
    block_num = 0  # Total number of blocks needed

    block_size = ifa_gqa_config.block_size
    block_table_batch = ifa_gqa_config.b
    max_num_blocks_per_query = math.ceil(max_s2 / s2_tile) * (s2_tile // block_size)
    block_table_shape = [block_table_batch, max_num_blocks_per_query]

    # Calculate number of blocks needed for each batch element
    for actual_seq in kv_actual_seqs:
        blocks_needed = math.ceil(actual_seq.item() / block_size)
        block_num_per_batch.append(blocks_needed)
        block_num += blocks_needed

    # Create all block indices and randomly permute them
    # This simulates non-contiguous physical memory allocation
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    # Create block table
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=device)
    block_idx = 0
    block_table_batch_idx = 0

    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1

    return block_table, block_num


def create_query_tensor(ifa_gqa_config: IfaGqaConfig, device: str) -> torch.Tensor:
    """Create query tensor with proper shape and initialization.

    Args:
        ifa_gqa_config: IFA GQA configuration.
        device: Device to create tensors on.

    Returns:
        Query tensor of shape [b, s1, n1, q_d].
    """
    query_shape = [ifa_gqa_config.b, ifa_gqa_config.n1,
                   ifa_gqa_config.s1, ifa_gqa_config.q_d]  # BSND
    query = torch.empty(query_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(
        device=device)
    return query


def create_kv_cache_tensors(
    ifa_gqa_config: IfaGqaConfig, kv_num_blocks: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create key and value cache tensors with proper initialization.

    Args:
        ifa_gqa_config: IFA GQA configuration.
        kv_num_blocks: kv cache num blocks
        device: Device to create tensors on.

    Returns:
        Tuple of (key_cache, value_cache) tensors.
    """
    # PA_BnNBsD format
    kv_cache_shape = [kv_num_blocks, ifa_gqa_config.n2, ifa_gqa_config.block_size, ifa_gqa_config.kv_d] 

    key_cache_fp32 = torch.empty(kv_cache_shape, dtype=torch.float32).uniform_(-1, 1)
    key_cache = key_cache_fp32.to(torch.float8_e4m3fn).to(device=device)
    value_cache_fp32 = torch.empty(kv_cache_shape, dtype=torch.float32).uniform_(-1, 1)
    value_cache = value_cache_fp32.to(torch.float8_e4m3fn).to(device=device)

    return key_cache, value_cache


def create_antiquant_scales(ifa_gqa_config: IfaGqaConfig, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create anti-quantization scale tensors.

    Args:
        ifa_gqa_config: IFA GQA configuration.
        device: Device to create tensors on.

    Returns:
        Tuple of (key_antiquant_scale, value_antiquant_scale) tensors.
    """
    antiquant_scale_shape = [ifa_gqa_config.n2, ifa_gqa_config.kv_d]

    key_antiquant_scale = torch.empty(antiquant_scale_shape, dtype=torch.bfloat16).uniform_(0, 0.2).to(device=device)
    value_antiquant_scale = torch.empty(antiquant_scale_shape, dtype=torch.bfloat16).uniform_(0, 0.2).to(device=device)

    return key_antiquant_scale, value_antiquant_scale


def log_input_info(inputs: Dict[str, torch.Tensor], ifa_gqa_config: IfaGqaConfig) -> None:
    """Log information about input tensors for debugging.

    Args:
        inputs: Dictionary containing all input tensors.
        ifa_gqa_config: IFA GQA configuration.
    """
    query = inputs['query']
    key = inputs['key']
    value = inputs['value']
    key_cache = inputs['key_cache']
    value_cache = inputs['value_cache']
    block_table = inputs['block_table']
    kv_actual_seqs = inputs['kv_actual_seqs']

    logger.info(f"query: shape {query.shape}, dtype {query.dtype}")
    logger.info(f"key: shape {key.shape}, dtype {key.dtype}")
    logger.info(f"key_cache: shape {key_cache.shape}, dtype {key_cache.dtype}")
    logger.info(f"value: shape {value.shape}, dtype {value.dtype}")
    logger.info(f"value_cache: shape {value_cache.shape}, dtype {value_cache.dtype}")
    logger.info(f"block_table: shape {block_table.shape}, dtype {block_table.dtype}")
    logger.info(f"kv_actual_seqs: shape {kv_actual_seqs.shape}, "
                f"dtype {kv_actual_seqs.dtype}, ele: {ifa_gqa_config.s2}")
    logger.info(f"block_size: {ifa_gqa_config.block_size}")
    logger.info(f"key_antiquant_scale: shape {inputs['key_antiquant_scale'].shape}, "
                f"dtype {inputs['key_antiquant_scale'].dtype}")
    logger.info(f"value_antiquant_scale: shape {inputs['value_antiquant_scale'].shape}, "
                f"dtype {inputs['value_antiquant_scale'].dtype}")


def gen_inputs(ifa_gqa_config: IfaGqaConfig, tile_config: AttentionTileConfig,
               device: str) -> Dict[str, torch.Tensor]:
    """Generate input tensors for IFA GQA computation.

    Args:
        ifa_gqa_config: IFA GQA configuration.
        device: Device to create tensors on.

    Returns:
        Dictionary containing all input tensors.
    """
    query = create_query_tensor(ifa_gqa_config, device)
    if isinstance(ifa_gqa_config.s2, list):
        max_s2 = max(ifa_gqa_config.s2)
        kv_actual_seqs = torch.tensor(ifa_gqa_config.s2, dtype=torch.int32, device=device)
    else:
        max_s2 = ifa_gqa_config.s2
        kv_actual_seqs = torch.tensor([ifa_gqa_config.s2] * ifa_gqa_config.b, dtype=torch.int32, device=device)
    
    s2_tile = tile_config.s2_tile
    block_table, kv_num_blocks = gen_block_table(ifa_gqa_config, kv_actual_seqs, max_s2, s2_tile, device)
    key_cache, value_cache = create_kv_cache_tensors(ifa_gqa_config, kv_num_blocks, device)

    key = kv_cache_concat(key_cache, kv_actual_seqs, block_table, ifa_gqa_config, max_s2, device)
    value = kv_cache_concat(value_cache, kv_actual_seqs, block_table, ifa_gqa_config, max_s2, device)

    key_antiquant_scale, value_antiquant_scale = create_antiquant_scales(ifa_gqa_config, device)

    inputs = {
        'query': query,
        'key': key,
        'value': value,
        'key_cache': key_cache,
        'value_cache': value_cache,
        'kv_actual_seqs': kv_actual_seqs,
        'block_table': block_table,
        'key_antiquant_scale': key_antiquant_scale,
        'value_antiquant_scale': value_antiquant_scale
    }

    log_input_info(inputs, ifa_gqa_config)

    return inputs


def antiquant_data(data: torch.Tensor,
                   antiquant_scale: torch.Tensor) -> torch.Tensor:
    """Apply anti-quantization to convert quantized data to floating point.

    Args:
        data: Quantized input tensor (already converted from float8_e4m3fn to bfloat16).
        antiquant_scale: Scale factor for de-quantization.

    Returns:
        De-quantized tensor in floating point format.
    """
    antiquant_scale_broadcast = antiquant_scale.reshape(1, antiquant_scale.shape[0], 1, antiquant_scale.shape[1])
    out_data = data * antiquant_scale_broadcast
    return out_data


def ifa_gqa_antiquant_golden(ifa_gqa_config: IfaGqaConfig,
                             ifa_gqa_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Golden reference implementation for IFA GQA with anti-quantization.
    
    Args:
        ifa_gqa_config: IFA GQA configuration parameters.
        ifa_gqa_inputs: Dictionary containing input tensors:
            - query: Query tensor of shape [b, s1, n1, q_d]
            - key: Key tensor of shape [b, n2, s2, kv_d]
            - value: Value tensor of shape [b, n2, s2, kv_d]
            - key_antiquant_scale: Scale for de-quantizing key
            - value_antiquant_scale: Scale for de-quantizing value

    Returns:
        Attention output tensor of shape [b, s1, n1, kv_d].
    """
    softmax_scale = ifa_gqa_config.softmax_scale
    n1 = ifa_gqa_config.n1
    n2 = ifa_gqa_config.n2
    s1 = ifa_gqa_config.s1
    group = n1 // n2

    query = ifa_gqa_inputs['query']
    key = antiquant_data(ifa_gqa_inputs['key'], ifa_gqa_inputs['key_antiquant_scale'])
    value = antiquant_data(ifa_gqa_inputs['value'], ifa_gqa_inputs['value_antiquant_scale'])
    kv_actual_seqs = ifa_gqa_inputs['kv_actual_seqs']
    b = query.shape[0]
    kv_d = key.shape[-1]

    logger.info(f"query: shape {query.shape}, dtype {query.dtype}")
    logger.info(f"key: shape {key.shape}, dtype {key.dtype}")
    logger.info(f"value: shape {value.shape}, dtype {value.dtype}")

    attention_out = torch.zeros([b, n1, s1, kv_d], dtype=torch.float32, device=query.device)

    for b_idx in range(b):
        total_kv_len = kv_actual_seqs[b_idx].item()

        for s1_idx in range(s1):
            cur_s2 = total_kv_len - s1 + 1 + s1_idx

            for n2_idx in range(n2):
                q_head_start = n2_idx * group
                q_head_end = (n2_idx + 1) * group
                q_cur = query[b_idx:b_idx + 1, q_head_start:q_head_end, s1_idx:s1_idx + 1, :]
                k_cur = key[b_idx:b_idx + 1, n2_idx:n2_idx + 1, :cur_s2, :]
                v_cur = value[b_idx:b_idx + 1, n2_idx:n2_idx + 1, :cur_s2, :]

                k_expanded = k_cur.repeat_interleave(group, dim=1)
                v_expanded = v_cur.repeat_interleave(group, dim=1)

                qk_mm_res = torch.matmul(q_cur, k_expanded.transpose(-2, -1))
                qk_ele_res = qk_mm_res * softmax_scale
                softmax_res = F.softmax(qk_ele_res, dim=-1)
                attn_group = torch.matmul(softmax_res, v_expanded)

                attention_out[b_idx, q_head_start:q_head_end, s1_idx:s1_idx + 1, :] = attn_group[0]

    attention_out = attention_out.to(query.dtype)
    logger.info(f"attention_out: shape{attention_out.shape}, dtype {attention_out.dtype}")
    return attention_out


def get_case_config(case_name):
    base_params = {"layout": "BNSD", "block_size": 128, "d": 128, "softmax_scale": 128 ** -0.5}
    if case_name.startswith("1b2k"):
        params = {"b": 1, "n1": 8, "s1": 1, "s2": 2 * 1024, "n2": 1}
    elif case_name.startswith("8b2kqs2"):
        params = {"b": 8, "n1": 8, "s1": 2, "s2": 2 * 1024, "n2": 1}
    elif case_name.startswith("16b4kqs3"):
        params = {"b": 16, "n1": 8, "s1": 3, "s2": 4 * 1024, "n2": 1}
    elif case_name.startswith("32b8k_d256"):
        params = {"b": 32, "n1": 8, "s1": 1, "s2": 8 * 1024, "n2": 1, "d": 256, "softmax_scale": 256 ** -0.5}
    elif case_name.startswith("64b2k_kvn2"):
        params = {"b": 64, "n1": 8, "s1": 1, "s2": 2 * 1024, "n2": 2}
    elif case_name.startswith("4b16k"):
        params = {"b": 4, "n1": 64, "s1": 2, "s2": 16 * 1024, "n2": 8}
    elif case_name.startswith("64b16k"):
        params = {"b": 64, "n1": 64, "s1": 2, "s2": 16 * 1024, "n2": 8}
    elif case_name.startswith("vary_s2_2b16k"):
        s2 = [16383, 16385]
        params = {"b": 2, "n1": 8, "s1": 1, "s2": s2, "n2": 1}
    elif case_name.startswith("vary_s2_4b2k"):
        s2 = [1024, 2048, 3072, 4096]
        params = {"b": 4, "n1": 8, "s1": 1, "s2": s2, "n2": 1}
    elif case_name.startswith("vary_s2_2b2k_kvn2"):
        s2 = [1536, 2560]
        params = {"b": 2, "n1": 8, "s1": 1, "s2": s2, "n2": 2}
    elif case_name.startswith("vary_s2_4b8k_qs2"):
        s2 = [4096, 6144, 10240, 12288]
        params = {"b": 4, "n1": 8, "s1": 2, "s2": s2, "n2": 1}
    elif case_name.startswith("vary_s2_2b4k_d256"):
        s2 = [3072, 5120]
        params = {"b": 2, "n1": 8, "s1": 1, "s2": s2, "n2": 1, "d": 256, "softmax_scale": 256 ** -0.5}
    elif case_name.startswith("vary_s2_4b8k_qs3_kvn2"):
        s2 = [2048, 4096, 8192, 16384]
        params = {"b": 4, "n1": 8, "s1": 3, "s2": s2, "n2": 2}

    base_params.update(params)

    case_config = IfaGqaConfig(layout=base_params["layout"], b=base_params["b"], n1=base_params["n1"], 
                               s1=base_params["s1"], q_d=base_params["d"], n2=base_params["n2"], 
                               s2=base_params["s2"], kv_d=base_params["d"], 
                               block_size=base_params["block_size"], softmax_scale=base_params["softmax_scale"])

    return case_config


def get_tile_config(case_config):
    n1 = case_config.n1
    n2 = case_config.n2
    group = n1 // n2

    s2 = case_config.s2
    if isinstance(s2, int):
        if s2 == 16 * 1024:
            unroll_list = [16, 8, 1]
        else:
            unroll_list = [8, 1]
    else:  # s2 is list
        s2_mean = int(sum(s2) // len(s2))
        if s2_mean == 16 * 1024:
            unroll_list = [16, 8, 1]
        else:
            unroll_list = [8, 1]

    g_tile = group

    if group in [64, 128]:
        g_tile = 8

    tile_config = AttentionTileConfig(
        g_tile=g_tile,
        s2_tile=512,
        v0_tile=[64, 128],
        c1_tile=[[16, 16], [128, 128], [256, 256]],
        v1_tile=[128, 512],
        c2_tile=[[16, 16], [256, 512], [128, 128]],
        v2_tile=[4, 128],
        unroll_list=unroll_list
    )
    return tile_config


def do_test_incre_flash_attention_gqa_antiquant(case_name: str) -> None:
    """Execute test for incremental flash attention GQA with anti-quantization.

    Args:
        case_name: Name identifier for the test case.
    """
    logger.info("*" * 60)
    logger.info(f"Run incre_flash_attention_gqa_antiquant {case_name} case")
    logger.info("*" * 60 + "\n")

    device = get_device()

    case_config = get_case_config(case_name)
    tile_config = get_tile_config(case_config)
    ifa_gqa_inputs = gen_inputs(case_config, tile_config, device)

    gqa_antiquant_golden = ifa_gqa_antiquant_golden(case_config, ifa_gqa_inputs)

    pypto_kernel_inputs = dict(
        query=ifa_gqa_inputs['query'],
        key=ifa_gqa_inputs['key_cache'],
        value=ifa_gqa_inputs['value_cache'],
        key_antiquant_scale=ifa_gqa_inputs['key_antiquant_scale'],
        value_antiquant_scale=ifa_gqa_inputs['value_antiquant_scale'],
        kv_actual_seqs=ifa_gqa_inputs['kv_actual_seqs'],
        block_table=ifa_gqa_inputs['block_table'],
        tile_config=tile_config
    )
    pypto_atten_out = incre_flash_attention_gqa_antiquant(**pypto_kernel_inputs)
    compare(pypto_atten_out.cpu(), gqa_antiquant_golden.cpu(), "pypto_atten_out",
            atol=0.0001, rtol=0.0078125, max_error_ratio=0.005)
    print("[PRECISION_PASS]")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_1b2k():
    do_test_incre_flash_attention_gqa_antiquant("1b2k")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_8b2kqs2():
    do_test_incre_flash_attention_gqa_antiquant("8b2kqs2")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_16b4kqs3():
    do_test_incre_flash_attention_gqa_antiquant("16b4kqs3")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_32b8k_d256():
    do_test_incre_flash_attention_gqa_antiquant("32b8k_d256")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_64b2k_kvn2():
    do_test_incre_flash_attention_gqa_antiquant("64b2k_kvn2")


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_incre_flash_attention_gqa_antiquant_4b16k():
    do_test_incre_flash_attention_gqa_antiquant("4b16k")


@pytest.mark.soc("950")
@pytest.mark.skip(reason="perf")
def test_incre_flash_attention_gqa_antiquant_64b16k():
    do_test_incre_flash_attention_gqa_antiquant("64b16k")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_2b16k():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_2b16k")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_4b2k():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_4b2k")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_2b2k_kvn2():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_2b2k_kvn2")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_4b8k_qs2():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_4b8k_qs2")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_2b4k_d256():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_2b4k_d256")


@pytest.mark.soc("950")
def test_incre_flash_attention_gqa_antiquant_vary_s2_4b8k_qs3_kvn2():
    do_test_incre_flash_attention_gqa_antiquant("vary_s2_4b8k_qs3_kvn2")


def main() -> None:
    """Main entry point for IFA GQA anti-quantization experimental."""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("PyPTO incre_flash_attention_gqa_antiquant experimental")
    logger.info("=" * 60 + "\n")

    test_incre_flash_attention_gqa_antiquant_1b2k()
    test_incre_flash_attention_gqa_antiquant_8b2kqs2()
    test_incre_flash_attention_gqa_antiquant_16b4kqs3()
    test_incre_flash_attention_gqa_antiquant_32b8k_d256()
    test_incre_flash_attention_gqa_antiquant_64b2k_kvn2()
    test_incre_flash_attention_gqa_antiquant_64b16k()
    test_incre_flash_attention_gqa_antiquant_4b16k()
    test_incre_flash_attention_gqa_antiquant_vary_s2_2b16k()
    test_incre_flash_attention_gqa_antiquant_vary_s2_4b2k()
    test_incre_flash_attention_gqa_antiquant_vary_s2_2b2k_kvn2()
    test_incre_flash_attention_gqa_antiquant_vary_s2_4b8k_qs2()
    test_incre_flash_attention_gqa_antiquant_vary_s2_2b4k_d256()
    test_incre_flash_attention_gqa_antiquant_vary_s2_4b8k_qs3_kvn2()

    logger.info("All test cases passed!")


if __name__ == "__main__":
    main()