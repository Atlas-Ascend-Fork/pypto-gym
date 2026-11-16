# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Autograd for the independent affine-summary CP KDA implementation."""

__all__ = ['chunk_kda']
import math
import torch
from torch.autograd.function import once_differentiable
try:
    from .flash_kda import flash_kda_stage1_wrapper, flash_kda_stage2_wrapper
    from .flash_kda_bwd import flash_kda_bwd_stage1_wrapper, flash_kda_bwd_stage2_wrapper
    from .kda_cp import field, prepare_cp_metadata, all_gather_summary, sum_parameter_gradients
except ImportError:
    from flash_kda import flash_kda_stage1_wrapper, flash_kda_stage2_wrapper
    from flash_kda_bwd import flash_kda_bwd_stage1_wrapper, flash_kda_bwd_stage2_wrapper
    from kda_cp import field, prepare_cp_metadata, all_gather_summary, sum_parameter_gradients


def _heads(x, start, length, layout, beta=False, state=False):
    if x is None:
        return None
    axis = 1 if state or layout == 'BNSD' else x.ndim - (1 if beta else 2)
    return x.narrow(axis, start, length)


def _buckets(q, k, v, g, beta, initial, a, bias, layout, meta, width):
    nk, nv = meta['global_Nk'], meta['global_Nv']
    ratio = nv // nk
    for ks in range(0, nk, width):
        kn = min(width, nk - ks)
        vs = ks * ratio
        vn = kn * ratio
        cp = dict(meta, hk_start=ks, hv_start=vs)
        raw = (_heads(q, ks, kn, layout), _heads(k, ks, kn, layout), _heads(v, vs, vn, layout),
             _heads(g, vs, vn, layout), _heads(beta, vs, vn, layout, beta=True))
        yield cp, raw, _heads(initial, vs, vn, layout, state=True), a[vs:vs + vn], bias[vs:vs + vn], ks, kn, vs, vn


class _KDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        initial,
        a,
        bias,
        scale,
        lower,
        layout,
        meta,
        group,
        backend,
        policy,
        cores,
        recompute,
        return_h,
        width,
    ):
        ctx.set_materialize_grads(False)
        out = torch.empty(tuple(v.shape), dtype=v.dtype, device=v.device)
        segments = len(meta['sequence_lengths']) if layout == 'TND' else q.shape[0]
        final = torch.empty((segments, meta['global_Nv'], 128, 128), dtype=torch.float32, device=v.device)
        saved = []
        for cp, raw, init, aa, bb, _, _, vs, vn in _buckets(q, k, v, g, beta, initial, a, bias, layout, meta, width):
            prepared, packet = flash_kda_stage1_wrapper(*raw, scale, aa, bb, lower, init,
                cp_meta=cp, layout_qkv=layout, is_training=True, num_cores=cores)
            all_packets = all_gather_summary(packet, cp, group, backend)
            _, _, cache = flash_kda_stage2_wrapper(prepared, all_packets, cp_meta=cp, num_cores=cores,
                out=_heads(out, vs, vn, layout), final_state_out=_heads(final, vs, vn, layout, state=True))
            saved.append(all_packets if recompute else cache)
        ctx.save_for_backward(q, k, v, g, beta, a, bias, *(() if initial is None else (initial,)))
        ctx.config = (scale, lower, layout, meta, group, backend, policy, cores, recompute, width)
        ctx.saved = saved
        if return_h:
            h = cache['h']
            ctx.mark_non_differentiable(h)
            return out, final, h
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, d_final, *unused):
        q, k, v, g, beta, a, bias, *state = ctx.saved_tensors
        initial = state[0] if state else None
        scale, lower, layout, meta, group, backend, policy, cores, recompute, width = ctx.config
        dq, dk, dv, dg, db = (torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device) for x in (q, k, v, g, beta))
        segments = len(meta['sequence_lengths']) if layout == 'TND' else q.shape[0]
        dh0 = torch.empty((segments, meta['global_Nv'], 128, 128), dtype=torch.float32, device=q.device)
        params = torch.empty((meta['global_Nv'] * 129,), dtype=torch.float32, device=q.device)
        for i, (cp, raw, init, aa, bb, ks, kn, vs, vn) in enumerate(
            _buckets(q, k, v, g, beta, initial, a, bias, layout, meta, width)
        ):
            cache = ctx.saved[i]
            if recompute:
                # Explicit option: two additional forward launches per bucket.
                prepared, _ = flash_kda_stage1_wrapper(*raw, scale, aa, bb, lower, init,
                    cp_meta=cp, layout_qkv=layout, is_training=True, num_cores=cores)
                _, _, cache = flash_kda_stage2_wrapper(prepared, ctx.saved[i], cp_meta=cp, num_cores=cores)
            dy = _heads(do, vs, vn, layout)
            seed = _heads(d_final, vs, vn, layout, state=True)
            summary, _ = flash_kda_bwd_stage1_wrapper(
                cache, dy, scale, seed, cp_meta=cp, num_cores=cores, param_grad_buffer=params)
            all_summary = all_gather_summary(summary, cp, group, backend)
            bindings = dict(dq=_heads(dq, ks, kn, layout), dk=_heads(dk, ks, kn, layout),
                dv=_heads(dv, vs, vn, layout), dbeta=_heads(db, vs, vn, layout, beta=True),
                dg=_heads(dg, vs, vn, layout), dh_initial_state=_heads(dh0, vs, vn, layout, state=True))
            flash_kda_bwd_stage2_wrapper(raw[0], raw[1], raw[3], aa, bb, cache, dy, all_summary, params,
                scale, lower, seed, cp_meta=cp, num_cores=cores, output_buffers=bindings)
        params = sum_parameter_gradients(params, meta, group, backend, policy)
        nv = a.numel()
        return (dq, dk, dv, dg, db, dh0 if initial is not None else None,
            params[:nv].view_as(a), params[nv:].view_as(bias),
            None, None, None, None, None, None, None, None, None, None, None)


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False, safe_gate=True, lower_bound=-1.0, disable_recompute=True,
    return_intermediate_states=False, state_v_first=True, cu_seqlens=None, cu_seqlens_cpu=None,
    cp_context=None, *, A_log, dt_bias, layout_qkv='BSND', num_cores=None, chunk_size=64, head_bucket_size=None):
    """Saved-cache path runs two kernels per direction; states are V-first.

    dt_bias is explicit FP32 input. Gloo is a validation-only CPU staging path.
    disable_recompute=False adds two forward launches during backward.
    """
    error = None
    if not all((use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, safe_gate, state_v_first)):
        error = 'CP v2 requires fused preprocessing, safe gate and V-first states'
    if allow_neg_eigval or chunk_size != 64:
        error = 'CP v2 supports beta_scale=1 and chunk_size=64'
    scale = 128**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or not -5 <= lower_bound <= 0:
        error = 'Invalid scale/lower_bound'
    if cu_seqlens is None:
        cu_seqlens = field(cp_context, 'cu_seqlens')
    packed_bsnd = cu_seqlens is not None and q.ndim == 4
    if packed_bsnd:
        if q.shape[0] != 1 or layout_qkv != 'BSND':
            raise ValueError('Packed BSND adapter requires B=1')
        q, k, v, g, beta = (x[0] for x in (q, k, v, g, beta))
        layout_qkv = 'TND'
    nk = q.shape[1] if layout_qkv == 'BNSD' else q.shape[-2]
    if return_intermediate_states and head_bucket_size is not None and int(head_bucket_size) < nk:
        error = 'return_intermediate_states requires one head bucket'
    meta, group, backend, policy = prepare_cp_metadata(q, v, layout_qkv, cp_context, cu_seqlens, scale, lower_bound,
        cu_seqlens_cpu, head_bucket_size, error)
    width = meta['global_Nk'] if head_bucket_size is None else int(head_bucket_size)
    if not 1 <= width <= meta['global_Nk']:
        raise ValueError('head_bucket_size counts complete key-head groups')
    if return_intermediate_states and width < meta['global_Nk']:
        raise ValueError('return_intermediate_states requires one head bucket')
    values = _KDA.apply(q, k, v, g, beta, initial_state, A_log, dt_bias, scale, float(lower_bound),
        layout_qkv, meta, group, backend, policy, num_cores, not disable_recompute, return_intermediate_states, width)
    out, final, *states = values
    if packed_bsnd:
        out = out.unsqueeze(0)
    result = (out, final if output_final_state else None)
    return result + tuple(states) if return_intermediate_states else result
