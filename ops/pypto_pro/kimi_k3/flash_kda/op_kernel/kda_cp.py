# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Affine-summary CP transport, with no sequential boundary handoff."""
from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class KDACPContext:
    group: object = None
    cu_seqlens: object = None
    cu_seqlens_cpu: object = None
    first_continues: object = None
    last_continues: object = None
    sequence_ids: object = None
    param_grad_reduce: str = 'sum'


def field(context, name, default=None):
    return context.get(name, default) if isinstance(context, dict) else getattr(context, name, default)


def prepare_cp_metadata(q, v, layout, context, cu_seqlens, scale, lower_bound, cu_seqlens_cpu=None,
                        head_bucket_size=None, preflight_error=None):
    """Explicit host metadata/H2D setup, outside all compute wrappers."""
    group = field(context, 'group')
    if context is not None and not dist.is_initialized():
        raise RuntimeError('CP requires an initialized process group')
    rank, size = (dist.get_rank(group), dist.get_world_size(group)) if context is not None else (0, 1)
    backend = str(dist.get_backend(group)).lower() if context is not None else 'local'
    if backend not in ('local', 'hccl', 'gloo'):
        raise ValueError('Use HCCL or validation-only CPU-staged Gloo')
    policy = field(context, 'param_grad_reduce', 'sum')
    packed = layout == 'TND'
    if packed:
        tokens, nk, d = q.shape
        nv = v.shape[1]
        b = 1
    elif layout == 'BSND':
        b, tokens, nk, d = q.shape
        nv = v.shape[2]
    elif layout == 'BNSD':
        b, nk, tokens, d = q.shape
        nv = v.shape[1]
    else:
        raise ValueError('layout must be BSND, BNSD, or TND')
    error = preflight_error
    if cu_seqlens is None:
        cu_seqlens = field(context, 'cu_seqlens')
    if packed:
        if cu_seqlens is None:
            error = 'TND requires cu_seqlens'
            bounds = (0, int(tokens))
        else:
            host = cu_seqlens_cpu if cu_seqlens_cpu is not None else field(context, 'cu_seqlens_cpu', cu_seqlens)
            if host is None:
                host = cu_seqlens
            bounds = tuple(int(x) for x in host.detach().cpu().tolist())
            if (
                len(bounds) < 2
                or bounds[0] != 0
                or bounds[-1] != tokens
                or any(a >= b for a, b in zip(bounds, bounds[1:]))
            ):
                error = 'CP boundaries must describe nonempty fragments'
                bounds = (0, max(1, int(tokens)))
        first = field(context, 'first_continues', False if size == 1 else None)
        last = field(context, 'last_continues', False if size == 1 else None)
        ids = field(context, 'sequence_ids')
        if ids is None and size == 1:
            ids = tuple(range(len(bounds) - 1))
        if first is None or last is None or ids is None or len(ids) != len(bounds) - 1:
            error = 'Packed CP requires boundary flags and sequence_ids'
            ids = tuple(range(len(bounds) - 1))
        if any(not isinstance(x, int) for x in ids):
            error = 'sequence_ids must contain integer logical IDs'
        elif len(set(ids)) != len(ids):
            error = 'Each local packed fragment must represent a distinct logical sequence'
        lengths = tuple(y - x for x, y in zip(bounds, bounds[1:]))
    else:
        first, last = rank > 0, rank + 1 < size
        lengths = (int(tokens),)
        ids = (0,)
    first, last = bool(first), bool(last)
    padded = sum((x + 63) // 64 * 64 for x in lengths)
    if policy not in ('sum', 'external'):
        error = 'param_grad_reduce must be sum or external'
    width = nk if head_bucket_size is None else int(head_bucket_size)
    if not 1 <= width <= nk:
        error = 'head_bucket_size must count complete key-head groups'
    if not tokens or padded > 131072 or not 1 <= nk <= nv <= 96 or nv % nk or d != 128:
        error = 'Invalid CP dimensions/padded token budget'
    if rank == 0 and first or rank == size - 1 and last:
        error = 'CP group edges cannot continue outside the group'
    if last and lengths[-1] % 64:
        error = 'Continuing fragment must end on a logical chunk boundary'
    local = dict(
        B=b,
        Nk=nk,
        Nv=nv,
        layout=layout,
        scale=float(scale),
        lower=float(lower_bound),
        policy=policy,
        width=width,
        first=first,
        last=last,
        head_id=ids[0],
        tail_id=ids[-1],
        ids=tuple(ids),
        error=error,
    )
    records = [local]
    if size > 1:
        records = [None] * size
        dist.all_gather_object(records, local, group=group)
    errors = [(i, x['error']) for i, x in enumerate(records) if x['error']]
    common = ('B', 'Nk', 'Nv', 'layout', 'scale', 'lower', 'policy', 'width')
    if any(any(x[k] != records[0][k] for k in common) for x in records):
        errors.append((-1, 'ranks disagree on configuration'))
    for i in range(size - 1):
        if records[i]['last'] != records[i + 1]['first']:
            errors.append((i, 'boundary continuation mismatch'))
        elif packed and records[i]['last'] and records[i]['tail_id'] != records[i + 1]['head_id']:
            errors.append((i, 'sequence identity mismatch'))
    if packed and not errors:
        seen = set()
        previous = None
        for rank_record, record in enumerate(records):
            for index, sid in enumerate(record['ids']):
                continuation = index == 0 and record['first']
                if sid in seen and not (continuation and sid == previous):
                    errors.append((rank_record, 'logical sequence IDs must occupy one contiguous interval'))
                seen.add(sid)
                previous = sid
    if errors:
        raise ValueError(f'Invalid collective CP metadata: {errors}')
    meta = dict(cp_rank=rank, cp_size=size, first_continues=first, last_continues=last,
        global_Nk=nk, global_Nv=nv, hk_start=0, hv_start=0)
    if packed:
        starts = []
        valid = []
        indices = []
        offsets = [0]
        for seg, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            for c, start in enumerate(range(begin, end, 64)):
                starts.append(start)
                valid.append(min(64, end - start))
                indices.append((seg, c))
            offsets.append(len(starts))

        def tensor(values):
            return torch.tensor(values, dtype=torch.int64, device=q.device)
        meta.update(sequence_lengths=lengths, sequence_ids=tuple(ids), cu_seqlens=tensor(bounds),
            chunk_starts=tensor(starts), chunk_lengths=tensor(valid), chunk_offsets=tensor(offsets),
            h_offsets=tensor([x + i for i, x in enumerate(offsets)]), chunk_indices=tensor(indices))
    return meta, group, backend, policy


def all_gather_summary(packet, meta, group, backend):
    if meta['cp_size'] == 1:
        return packet
    shape = (meta['cp_size'] * packet.shape[0], *packet.shape[1:])
    if backend == 'hccl':
        gathered = torch.empty(shape, dtype=packet.dtype, device=packet.device)
        dist.all_gather_into_tensor(gathered, packet, group=group)
        return gathered
    local = packet.cpu()
    gathered = torch.empty(shape, dtype=packet.dtype, device='cpu')
    dist.all_gather_into_tensor(gathered, local, group=group)
    return gathered.to(packet.device)


def sum_parameter_gradients(buffer, meta, group, backend, policy):
    if meta['cp_size'] == 1 or policy == 'external':
        return buffer
    if backend == 'hccl':
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=group)
        return buffer
    staged = buffer.cpu()
    dist.all_reduce(staged, op=dist.ReduceOp.SUM, group=group)
    return staged.to(buffer.device)
