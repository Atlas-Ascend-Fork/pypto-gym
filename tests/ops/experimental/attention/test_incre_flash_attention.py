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

import logging
import math
import os
import sys
from dataclasses import dataclass, replace

import torch
import torch_npu

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

import pytest
import numpy as np
import torch.nn.functional as F
from experimental.attention.incre_flash_attention_impl import incre_flash_attention
from numpy.testing import assert_allclose

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
# Set up logging format with timestamp, level, filename, and line number
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


@dataclass
class AttentionConfig:
    """
    Configuration parameters for attention computation.

    Attributes:
        b: Batch size
        s1: Query sequence length
        s2: Key/Value sequence length (maximum)
        n1: Number of query heads
        n2: Number of key/value heads (for grouped query attention)
        q_d: Query head dimension
        kv_d: Key/Value head dimension
        block_size: Size of each block in paged KV cache (default: 128)
        max_num_blocks_per_query: Maximum number of blocks per query sequence
        softmax_scale: Scaling factor for softmax (default: 1.0)
        kv_actual_seqs: Tensor containing actual sequence lengths for each batch
        block_table_batch: Batch size for block table
        kv_num_blocks: Total number of KV blocks
    """
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_actual_seqs: torch.Tensor = None
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def gen_block_table(atten_cfg: AttentionConfig, device: str):
    """
    Generate a block table for paged KV cache.

    The block table maps logical block indices to physical block indices,
    enabling non-contiguous memory access patterns. This is essential for
    efficient memory management in autoregressive generation.

    Args:
        atten_cfg: Attention configuration containing sequence lengths and block settings
        device: Device to create tensors on (e.g., 'npu:0', 'cpu')

    Returns:
        torch: Block table tensor of shape [batch_size, max_blocks_per_query]
               Contains physical block indices, or -1 for invalid blocks
    """
    block_num_per_batch = []
    block_num = 0  # Total number of blocks needed

    actual_seq_len = atten_cfg.kv_actual_seqs
    block_size = atten_cfg.block_size
    block_table_batch = atten_cfg.block_table_batch
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    block_table_shape = [block_table_batch, max_num_blocks_per_query]

    # Move to CPU if necessary for computation
    if actual_seq_len.device.type != 'cpu':
        actual_seq_len_cpu = actual_seq_len.cpu()
    else:
        actual_seq_len_cpu = actual_seq_len

    # Calculate number of blocks needed for each batch element
    for actual_seq in actual_seq_len_cpu:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

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
    return block_table


def kv_cache_concat(k_cache, v_cache, block_table, atten_cfg, device: str):
    """
    Concatenate KV cache blocks into contiguous tensors.

    This function reconstructs the full KV tensors from paged blocks
    using the block table. This is primarily used for reference/golden
    computation and verification.

    Args:
        k_cache: Key cache tensor of shape [num_blocks, n2, block_size, kv_d]
        v_cache: Value cache tensor of shape [num_blocks, n2, block_size, kv_d]
        block_table: Block table mapping logical to physical indices
        atten_cfg: Attention configuration
        device: Device for output tensors

    Returns:
        tuple: (k, v) where:
            - k: Contiguous key tensor of shape [b, n2, kv_max, kv_d]
            - v: Contiguous value tensor of shape [b, n2, kv_max, kv_d]
    """
    b = atten_cfg.b
    n2 = atten_cfg.n2
    kv_lora_rank = atten_cfg.q_d
    rope_dim = atten_cfg.kv_d
    block_size = atten_cfg.block_size
    kv_actual_seqs = atten_cfg.kv_actual_seqs
    dtype = v_cache.dtype

    # Move to CPU if necessary
    if kv_actual_seqs.device.type != 'cpu':
        kv_actual_seqs_cpu = kv_actual_seqs.cpu()
    else:
        kv_actual_seqs_cpu = kv_actual_seqs

    # Calculate maximum sequence length (padded to block size)
    kv_act_seq_max = torch.max(kv_actual_seqs_cpu).item()
    kv_max = math.ceil(kv_act_seq_max / block_size) * block_size

    # Initializes output tensors
    k = torch.zeros([b, n2, kv_max, kv_lora_rank], dtype=dtype, device=device)
    v = torch.zeros([b, n2, kv_max, rope_dim], dtype=dtype, device=device)

    # Reconstruct tensors by following block table
    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, n2, kv_max, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, n2, kv_max, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        # Copy blocks according to block table
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            # Copy block from cache to temp tensor
            kv_nope_temp_tensor[:, :, start_idx:end_idx, :] = v_cache[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, :, start_idx:end_idx, :] = k_cache[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        # Store result
        v[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k, v


def get_env_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def get_device(device_id: int = None, run_mode: str = "npu"):
    """
    Get the appropriate device string for computation.

    Args:
        device_id: Explicit device ID (optional)
        run_mode: Execution mode - "npu" for hardware, "sim" for simulation

    Returns:
        str: Device string (e.g., "npu:0" or "cpu")
    """
    if device_id is not None:
        cur_device_id = device_id
    else:
        cur_device_id = get_env_device_id()

    torch.npu.set_device(int(cur_device_id))
    device = f"npu:{cur_device_id}" if (run_mode == "npu" and cur_device_id is not None) else "cpu"
    return device


def get_base_params(case_name: str):
    """
    Get base parameters for predefined test cases.

    This function maps case names to their corresponding parameter sets.
    Case names follow the pattern: {batch_size}b{sequence_length}k
    For example: "1b16k" means batch_size=1, sequence_length=16*1024

    Args:
        case_name: Name of the test case

    Returns:
        dict: Dictionary containing parameters (b, s2, s1, d, n1, n2)

    Raises:
        Exception: If case_name is not recognized
    """
    params = {}
    if case_name.startswith("1b16k"):
        params = {"b": 1, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("2b16k"):
        params = {"b": 2, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("4b16k"):
        params = {"b": 4, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("8b16k"):
        params = {"b": 8, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("16b16k"):
        params = {"b": 16, "s2": 16 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("1b8k"):
        params = {"b": 1, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("2b8k"):
        params = {"b": 2, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("4b8k"):
        params = {"b": 4, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("8b8k"):
        params = {"b": 8, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    elif case_name.startswith("16b8k"):
        params = {"b": 16, "s2": 8 * 1024, "s1": 1, "d": 128, "n1": 12, "n2": 1}
    else:
        raise Exception(f"Case {case_name} does not exist.")
    return params


def get_ifa_atten_cfg(device: str, case_name: str):
    """
    Get attention configuration for a test case.

    This function creates a complete AttentionConfig for a given test case.

    Args:
        device: Device to create tensors on
        case_name: Name of the test case

    Returns:
        AttentionConfig: Configured attention parameters
    """
    base_params = get_base_params(case_name)
    b = base_params.get("b")
    s2 = base_params.get("s2")
    s1 = base_params.get("s1")
    d = base_params.get("d")
    n1 = base_params.get("n1")
    n2 = base_params.get("n2")

    # Calculate softmax scale (1 / sqrt(d))
    softmax_scale = d ** -0.5

    # Block table configuration
    block_table_batch = b
    block_size = 128
    max_num_blocks_per_query = math.ceil(s2 / block_size)
    kv_num_blocks = b * max_num_blocks_per_query

    # Create actual sequence lengths tensor (all sequences have full length)
    kv_actual_seqs = torch.tensor([s2] * b, dtype=torch.int32, device=device)

    # Create attention configuration
    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=n1, n2=n2, q_d=d, kv_d=d,
                                block_size=block_size,
                                max_num_blocks_per_query=max_num_blocks_per_query,
                                softmax_scale=softmax_scale, kv_actual_seqs=kv_actual_seqs,
                                block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks)
    return atten_cfg


def gen_qkv(atten_cfg: AttentionConfig, device: str):
    """
    Generate random query, key, and value tensors.

    Args:
        atten_cfg: Attention configuration
        device: Device to create tensors

    Returns:
        tuple: (q, k, v) where:
            - q: Query tensor of shape [b*s1, n1, q_d]
            - k: Key cache of shape [kv_num_blocks, n2, block_size, kv_d]
            - v: Value cache of shape [kv_num_blocks, n2, block_size, kv_d]
    """
    # Get query shape
    b = atten_cfg.b
    s1 = atten_cfg.s1
    n1 = atten_cfg.n1
    q_d = atten_cfg.q_d
    q_shape = [b * s1, n1, q_d]  # TND format

    # Get kv shape
    kv_num_blocks = atten_cfg.kv_num_blocks
    block_size = atten_cfg.block_size
    n2 = atten_cfg.n2
    kv_d = atten_cfg.kv_d
    kv_shape = [kv_num_blocks, n2, block_size, kv_d]  # PA_BnNBsD format

    # Generate random tensors with uniform distribution in [-1, 1]
    dtype = torch.bfloat16
    q = torch.empty(q_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    return q, k, v


def gen_ifa_golden(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, atten_cfg: AttentionConfig):
    """
    Generate golden (reference) output using PyTorch.

    Args:
        q: Query tensor of shape [b*s1, n1, d]
        k: Key tensor of shape [kv_num_blocks, n2, block_size, kv_d]
        v: Value tensor of shape [kv_num_blocks, n2, block_size, kv_d]
        atten_cfg: Attention configuration

    Returns:
        torch: Attention output tensor of shape [b*s1, n1, d]
    """
    b = atten_cfg.b
    s1 = atten_cfg.s1
    n2 = atten_cfg.n2
    d = atten_cfg.q_d
    softmax_scale = atten_cfg.softmax_scale
    kv_actual_seqs = atten_cfg.kv_actual_seqs

    atten_out = torch.zeros_like(q)

    # Compute attention for each batch, query position, and head
    for b_idx in range(b):
        for s1_idx in range(s1):
            for n2_idx in range(n2):
                # Get actual sequence length for this batch
                kv_seq_len = kv_actual_seqs[b_idx].item()

                # Calculate effective sequence length
                cur_s1_len = s1 - 1 - s1_idx
                seq_len = kv_seq_len - cur_s1_len

                # Extract query, key, and value for current position
                q_bs = q[b_idx * s1 + s1_idx]
                k_bs = k[b_idx, n2_idx:n2_idx + 1, :seq_len].reshape(seq_len, d)
                v_bs = v[b_idx, n2_idx:n2_idx + 1, :seq_len].reshape(seq_len, d)

                # First matrix multiplication: Q x K^T
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                qk_ele_res = qk_bmm_res * softmax_scale

                # Softmax computation
                softmax_res = F.softmax(qk_ele_res)

                # Second matrix multiplication: Softmax x V
                bmm2_res = torch.matmul(softmax_res, v_bs)

                # Store result
                atten_out[b_idx * s1 + s1_idx] = bmm2_res
    return atten_out


def do_test_incre_flash_attention(case_name: str):
    """
    Test the incremental flash attention implementation.

    This function runs a complete test for a given case:
    1. Generate test data
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO IFA
    4. Compare results

    Args:
        case_name: Name of the test case (e.g., "1b16k", "8b8k")
    """
    logger.info("*" * 60)
    logger.info(f"Run incre_flash_attention {case_name} case")
    logger.info("*" * 60 + "\n")

    device = get_device()
    atten_cfg = get_ifa_atten_cfg(device, case_name)

    q, k_cache, v_cache = gen_qkv(atten_cfg, device)
    logger.info(f"q.shape: {q.shape}")
    logger.info(f"k_cache.shape: {k_cache.shape}")
    logger.info(f"v_cache.shape: {v_cache.shape}")
    block_table = gen_block_table(atten_cfg, device)
    logger.info(f"block_table.shape: {block_table.shape}")
    k, v = kv_cache_concat(k_cache, v_cache, block_table, atten_cfg, device)
    logger.info(f"k.shape: {k.shape}")
    logger.info(f"v.shape: {v.shape}")

    kv_actual_seqs = atten_cfg.kv_actual_seqs
    logger.info(f"kv_actual_seqs: {kv_actual_seqs}")
    ifa_golden = gen_ifa_golden(q, k, v, atten_cfg)

    inputs = dict(
        query=q,
        key=k_cache,
        value=v_cache,
        block_table=block_table,
        actual_seq_lengths=kv_actual_seqs,
    )
    ifa_pypto_out = incre_flash_attention(**inputs)

    assert_allclose(np.array(ifa_golden.cpu().flatten().tolist()),
                    np.array(ifa_pypto_out.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)


def main():
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("PyPTO incre_flash_attention Example")
    logger.info("=" * 60 + "\n")

    # test incre_flash_attention kvs 16k
    test_incre_flash_attention_1b16k()
    test_incre_flash_attention_2b16k()
    test_incre_flash_attention_4b16k()
    test_incre_flash_attention_8b16k()
    test_incre_flash_attention_16b16k()

    # test incre_flash_attention kvs 8k
    test_incre_flash_attention_1b8k()
    test_incre_flash_attention_2b8k()
    test_incre_flash_attention_4b8k()
    test_incre_flash_attention_8b8k()
    test_incre_flash_attention_16b8k()


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_1b16k():
    do_test_incre_flash_attention("1b16k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_2b16k():
    do_test_incre_flash_attention("2b16k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_4b16k():
    do_test_incre_flash_attention("4b16k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_8b16k():
    do_test_incre_flash_attention("8b16k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_16b16k():
    do_test_incre_flash_attention("16b16k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_1b8k():
    do_test_incre_flash_attention("1b8k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_2b8k():
    do_test_incre_flash_attention("2b8k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_4b8k():
    do_test_incre_flash_attention("4b8k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_8b8k():
    do_test_incre_flash_attention("8b8k")


@pytest.mark.skip(reason="large test case")
def test_incre_flash_attention_16b8k():
    do_test_incre_flash_attention("16b8k")


if __name__ == "__main__":
    main()
