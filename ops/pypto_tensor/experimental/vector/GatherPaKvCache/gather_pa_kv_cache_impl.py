# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import collections

import pypto
import torch

INT32_MAX = 2**31 - 1
DEFAULT_GATHER_TILE_CONFIG = [16, 32]
LARGE_TOKEN_GATHER_TILE_CONFIG = [8, 8]

GatherRequest = collections.namedtuple(
    "GatherRequest",
    [
        "key_cache",
        "value_cache",
        "block_tables",
        "seq_lens",
        "key_ref",
        "value_ref",
        "seq_offset",
        "cache_mode",
        "is_seq_lens_cumsum",
        "run_mode",
    ],
)

GatherRefs = collections.namedtuple(
    "GatherRefs",
    ["key_ref", "value_ref", "seq_offset", "cache_mode", "is_seq_lens_cumsum", "run_mode"],
)


GatherInputs = collections.namedtuple(
    "GatherInputs",
    ["key_cache", "value_cache", "block_tables", "seq_lens"],
)


def _pop_seq_lens_arg(args: tuple, kwargs: dict) -> tuple[torch.Tensor, tuple]:
    if args:
        return args[0], args[1:]
    if "seq_lens" in kwargs:
        return kwargs.pop("seq_lens"), args
    raise TypeError("missing required argument: seq_lens")


def _parse_gather_refs(args: tuple, kwargs: dict, defaults: tuple) -> GatherRefs:
    names = ["key_ref", "value_ref", "seq_offset", "cache_mode", "is_seq_lens_cumsum", "run_mode"]
    if len(args) > len(names):
        raise TypeError("too many positional arguments")
    values = dict(zip(names, defaults))
    for name, value in zip(names, args):
        values[name] = value
    for name in names:
        if name in kwargs:
            values[name] = kwargs.pop(name)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")
    return GatherRefs(*(values[name] for name in names))


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "device_sched_mode": 1,
        "stitch_function_max_num": 128,
        "ready_on_host_tensors": ["block_tables", "seq_lens", "seq_offset"],
        "valid_shape_optimize": 1,
    },
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
)
def _gather_pa_kv_cache_nd_kernel_npu(
    key_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    block_tables: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    seq_lens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    key_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    seq_offset: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    tile_config: list,
):
    block_size = key_cache.shape[1]
    num_blocks = key_cache.shape[0]
    key_num_heads = key_cache.shape[2]
    key_dim = key_cache.shape[3]
    value_num_heads = value_cache.shape[2]
    value_dim = value_cache.shape[3]
    key_cache_3d = pypto.reshape(key_cache, [num_blocks * block_size, key_num_heads, key_dim], inplace=True)
    value_cache_3d = pypto.reshape(value_cache, [num_blocks * block_size, value_num_heads, value_dim], inplace=True)

    for q_base, q_unroll in pypto.loop_unroll(
        0, block_tables.shape[0], 1, name="gather_q_loop", idx_name="q_idx", unroll_list=[2, 1]
    ):
        for q_inner in range(q_unroll):
            q_idx = q_base + q_inner
            out_start = seq_lens[q_idx]
            seq_len = seq_lens[q_idx + 1] - out_start
            table_offset = seq_offset[q_idx] // block_size
            block_count = pypto.ceildiv(seq_len, block_size)

            for block_idx in pypto.loop(block_count, name="gather_block_loop", idx_name="block_idx", unroll_list=[1]):
                physical_block = block_tables[q_idx, table_offset + block_idx]
                token_offset = block_idx * block_size
                valid_tokens = (seq_len - token_offset).min(block_size)
                out_offset = out_start + token_offset
                cache_offset = physical_block * block_size
                key_shape = [block_size, key_num_heads, key_dim]
                value_shape = [block_size, value_num_heads, value_dim]
                key_valid = [valid_tokens, key_num_heads, key_dim]
                value_valid = [valid_tokens, value_num_heads, value_dim]

                pypto.set_vec_tile_shapes(tile_config[0], key_num_heads, key_dim)
                key_tile = pypto.view(key_cache_3d, key_shape, [cache_offset, 0, 0], valid_shape=key_valid)
                pypto.assemble(key_tile, [out_offset, 0, 0], key_ref)
                pypto.set_vec_tile_shapes(tile_config[1], value_num_heads, value_dim)
                value_tile = pypto.view(value_cache_3d, value_shape, [cache_offset, 0, 0], valid_shape=value_valid)
                pypto.assemble(value_tile, [out_offset, 0, 0], value_ref)


def _require_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


CachePairValidation = collections.namedtuple(
    "CachePairValidation",
    ["key_num_blocks", "key_block_size", "key_num_heads", "key_dim",
     "value_num_heads", "value_dim", "device"],
)


def _validate_cache_pair(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[int, int, int, int, int, int, torch.device]:
    _require_tensor("key_cache", key_cache)
    _require_tensor("value_cache", value_cache)
    if key_cache.dtype != torch.bfloat16:
        raise TypeError("key_cache must have dtype torch.bfloat16")
    if value_cache.dtype != torch.bfloat16:
        raise TypeError("value_cache must have dtype torch.bfloat16")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("key_cache/value_cache must have rank 4")
    if key_cache.device != value_cache.device:
        raise ValueError("key_cache and value_cache must be on the same device")

    key_num_blocks, key_block_size, key_num_heads, key_dim = key_cache.shape
    value_num_blocks, value_block_size, value_num_heads, value_dim = value_cache.shape
    if key_num_blocks != value_num_blocks or key_block_size != value_block_size:
        raise ValueError(
            "key_cache and value_cache must share num_blocks and block_size"
        )
    if key_num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if key_block_size <= 0:
        raise ValueError("block_size must be positive")
    if key_num_heads <= 0 or value_num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("head dimensions must be positive")
    return CachePairValidation(
        int(key_num_blocks),
        int(key_block_size),
        int(key_num_heads),
        int(key_dim),
        int(value_num_heads),
        int(value_dim),
        key_cache.device,
    )


def _validate_index_tensor(name: str, tensor: torch.Tensor, dim: int) -> None:
    _require_tensor(name, tensor)
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.dim() != dim:
        raise ValueError(f"{name} must have rank {dim}, got {tuple(tensor.shape)}")


def _build_seq_lens_cumsum(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_index_tensor("seq_lens", seq_lens, 1)
    if not is_seq_lens_cumsum:
        raise ValueError("only cumsum seq_lens [Q + 1] is supported")
    seq_lens_cpu = seq_lens.detach().cpu().contiguous().to(torch.int64)
    if seq_lens_cpu.numel() != q_count + 1:
        raise ValueError("seq_lens must have shape [Q + 1]")
    if int(seq_lens_cpu[0].item()) != 0:
        raise ValueError("cumsum seq_lens must start with 0")
    lengths = seq_lens_cpu[1:] - seq_lens_cpu[:-1]
    seq_lens_cumsum = seq_lens_cpu
    if bool((lengths < 0).any().item()):
        raise ValueError("seq_lens must describe non-negative lengths")
    total_tokens = int(seq_lens_cumsum[-1].item())
    if total_tokens < 0 or total_tokens > INT32_MAX:
        raise ValueError("total token count must fit in int32")
    return seq_lens_cumsum.to(torch.int32), lengths, total_tokens


def _get_seq_lens_meta(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq_lens_cumsum, lengths, total_tokens = _build_seq_lens_cumsum(
        seq_lens,
        q_count,
        is_seq_lens_cumsum,
    )
    return seq_lens_cumsum.to(device).contiguous(), lengths, total_tokens


def _checked_ref(ref: torch.Tensor | None, shape: tuple, device: torch.device, name: str) -> torch.Tensor:
    if ref is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    if ref.shape != shape or ref.dtype != torch.bfloat16 or ref.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype bf16, and device {device}")
    return ref


def _validate_seq_offsets(seq_offset: torch.Tensor, block_size: int, q_count: int) -> torch.Tensor:
    if seq_offset.numel() != q_count:
        raise ValueError("seq_offset must have shape [Q]")
    seq_offset_cpu = seq_offset.detach().cpu().contiguous().to(torch.int64)
    if bool((seq_offset_cpu < 0).any().item()):
        raise ValueError("seq_offset values must be non-negative")
    if bool((seq_offset_cpu % block_size != 0).any().item()):
        raise ValueError("seq_offset values must be divisible by block_size")
    return seq_offset_cpu


def _validate_block_tables(
    block_tables: torch.Tensor,
    lengths: torch.Tensor,
    table_offsets: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> None:
    block_tables_cpu = block_tables.detach().cpu().contiguous().to(torch.int64)
    for q_idx in range(int(block_tables.shape[0])):
        required_blocks = int((int(lengths[q_idx].item()) + block_size - 1) // block_size)
        table_end = int(table_offsets[q_idx].item()) + required_blocks
        if table_end > block_tables.shape[1]:
            raise ValueError(f"block_tables row {q_idx} is too short")
        if table_end > 0:
            start = int(table_offsets[q_idx].item())
            used_blocks = block_tables_cpu[q_idx, start:table_end]
            if bool((used_blocks < 0).any().item()) or bool((used_blocks >= num_blocks).any().item()):
                raise ValueError(f"block_tables row {q_idx} contains invalid block ids")


def _prepare_outputs_and_validate(request: GatherRequest) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if request.cache_mode != "Norm":
        raise NotImplementedError("network sweep path only supports cache_mode='Norm'")
    if request.run_mode != "npu":
        raise ValueError("only run_mode='npu' is supported")

    cache_meta = _validate_cache_pair(request.key_cache, request.value_cache)
    num_blocks, block_size, key_num_heads, key_dim, value_num_heads, value_dim, device = cache_meta
    _validate_index_tensor("block_tables", request.block_tables, 2)
    _validate_index_tensor("seq_offset", request.seq_offset, 1)
    q_count = int(request.block_tables.shape[0])
    if q_count <= 0:
        raise ValueError("block_tables first dimension must be positive")
    if request.block_tables.shape[1] <= 0:
        raise ValueError("block_tables second dimension must be positive")

    seq_lens_cumsum, lengths, total_tokens = _get_seq_lens_meta(
        request.seq_lens,
        q_count,
        request.is_seq_lens_cumsum,
        device,
    )
    key_ref = _checked_ref(request.key_ref, (total_tokens, key_num_heads, key_dim), device, "key_ref")
    value_ref = _checked_ref(request.value_ref, (total_tokens, value_num_heads, value_dim), device, "value_ref")
    seq_offset_cpu = _validate_seq_offsets(request.seq_offset, block_size, q_count)
    _validate_block_tables(request.block_tables, lengths, seq_offset_cpu // block_size, block_size, num_blocks)
    return key_ref, value_ref, seq_lens_cumsum, total_tokens


def _make_gather_request(inputs: GatherInputs, refs: GatherRefs, seq_offset: torch.Tensor) -> GatherRequest:
    return GatherRequest(
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_tables,
        inputs.seq_lens,
        refs.key_ref,
        refs.value_ref,
        seq_offset,
        refs.cache_mode,
        refs.is_seq_lens_cumsum,
        refs.run_mode,
    )


def _select_gather_tile_config(key_num_heads: int, key_dim: int, value_num_heads: int, value_dim: int) -> list[int]:
    if key_num_heads * key_dim >= 4096 or value_num_heads * value_dim >= 4096:
        return LARGE_TOKEN_GATHER_TILE_CONFIG
    return DEFAULT_GATHER_TILE_CONFIG


def gather_pa_kv_cache_out(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_lens, args = _pop_seq_lens_arg(args, kwargs)
    (
        _,
        _,
        key_num_heads,
        key_dim,
        value_num_heads,
        value_dim,
        device,
    ) = _validate_cache_pair(key_cache, value_cache)
    refs = _parse_gather_refs(args, kwargs, (None, None, None, "Norm", True, "npu"))
    seq_offset = refs.seq_offset
    if seq_offset is None:
        q_count = int(block_tables.shape[0])
        seq_offset = torch.zeros((q_count,), dtype=torch.int32, device=device)
    request = _make_gather_request(GatherInputs(key_cache, value_cache, block_tables, seq_lens), refs, seq_offset)
    key_ref, value_ref, seq_lens_cumsum, total_tokens = _prepare_outputs_and_validate(request)
    if total_tokens == 0:
        return key_ref, value_ref

    tile_config = _select_gather_tile_config(key_num_heads, key_dim, value_num_heads, value_dim)

    _gather_pa_kv_cache_nd_kernel_npu(
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_tables.contiguous(),
        seq_lens_cumsum,
        key_ref,
        value_ref,
        seq_offset.contiguous(),
        list(tile_config),
    )
    return key_ref, value_ref


def gather_pa_kv_cache_wrapper(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_lens, args = _pop_seq_lens_arg(args, kwargs)
    refs = _parse_gather_refs(args, kwargs, (None, None, None, "Norm", True, "npu"))
    _, _, _, _, _, _, device = _validate_cache_pair(key_cache, value_cache)
    seq_offset = refs.seq_offset
    if seq_offset is None:
        q_count = int(block_tables.shape[0])
        seq_offset = torch.zeros((q_count,), dtype=torch.int32, device=device)
    request = _make_gather_request(GatherInputs(key_cache, value_cache, block_tables, seq_lens), refs, seq_offset)
    key_ref, value_ref, _, total_tokens = _prepare_outputs_and_validate(request)
    if total_tokens == 0:
        return key_ref, value_ref
    return gather_pa_kv_cache_out(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        key_ref,
        value_ref,
        seq_offset,
        cache_mode=refs.cache_mode,
        is_seq_lens_cumsum=refs.is_seq_lens_cumsum,
        run_mode=refs.run_mode,
    )
