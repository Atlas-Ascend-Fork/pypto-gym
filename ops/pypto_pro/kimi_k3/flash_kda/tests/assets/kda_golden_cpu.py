# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU references for Kimi K3 KDA tests.

golden(): raw-input FP32 recurrent forward, differentiated by PyTorch autograd.
stable(): cache-conditioned chunk backward, with FP32 or BF16-rounded operands.
The latter checks the backward cache contract; it is not an independent forward
Golden. The third reference rounds operands to BF16, then performs FP32 matmul
on CPU. No reference imports the device kernels or runs NPU arithmetic.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _block(q, k, v, g, beta, state, scale):
    outputs = []
    for i in range(q.shape[1]):
        state = state * g[:, i, :, :, None].exp()
        innovation = v[:, i] - (k[:, i, :, :, None] * state).sum(-2)
        state = state + torch.einsum("bhk,bhv->bhkv", beta[:, i, :, None] * k[:, i], innovation)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * scale, state))
    return torch.stack(outputs, 1), state


def golden(q, k, v, g, beta, a, bias, initial, cfg, bounds):
    if cfg.get("use_qk_l2norm_in_kernel"):
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = (x.repeat_interleave(v.shape[2] // x.shape[2], 2) for x in (q, k))
    if cfg.get("use_gate_in_kernel"):
        x = g + bias.reshape(v.shape[2], 128)
        alpha = a.reshape(v.shape[2], 1).exp()
        lb = cfg.get("lower_bound")
        g = -alpha * F.softplus(x) if lb is None else lb * torch.sigmoid(alpha * x)
    if cfg.get("use_beta_sigmoid_in_kernel"):
        beta = beta.sigmoid() * (2 if cfg.get("allow_neg_eigval") else 1)
    if cfg.get("state_v_first"):
        initial = initial.transpose(-1, -2)
    outputs, finals, all_entries = [], [], []
    seqs = (
        [(i, 0, q.shape[1]) for i in range(q.shape[0])]
        if bounds is None
        else [(0, x, y) for x, y in zip(bounds, bounds[1:])]
    )
    for seq, (batch, start, end) in enumerate(seqs):
        state, parts, entries = initial[seq: seq + 1], [], []
        for off in range(start, end, 64):
            entries.append(state.detach())
            xs = [x[batch: batch + 1, off: min(off + 64, end)] for x in (q, k, v, g, beta)]
            o, state = checkpoint(
                _block, *xs, state, cfg.get("scale", 128**-0.5), use_reentrant=True
            )
            parts.append(o)
        outputs.append(torch.cat(parts, 1) if parts else v[batch: batch + 1, start:end])
        finals.append(state)
        all_entries.append(
            torch.stack(entries, 1) if entries else initial.new_empty(1, 0, v.shape[2], 128, 128)
        )
    o = torch.cat(outputs, 0 if bounds is None else 1)
    ht = torch.cat(finals, 0)
    result = (o, ht.transpose(-1, -2) if cfg.get("state_v_first") else ht)
    if cfg.get("return_intermediate_states"):
        states = torch.cat(all_entries, 0 if bounds is None else 1)
        return result + (states.transpose(-1, -2) if cfg.get("state_v_first") else states,)
    return result


def activation_reference(core, raw, a, bias, lower_bound, length):
    """CPU expressions independent of the kernel helpers; no NPU arithmetic."""
    dq, dk, dv, db, dg = (v[:, :length].float() for v in core)
    q, k, g, beta = (x.float() for x in raw)

    def norm(x, d):
        denom = (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        return d / denom - x * (d * x).sum(-1, keepdim=True) / denom.pow(3)

    av = a.exp().reshape(-1, 1)
    x = g + bias
    sig = torch.sigmoid(av * x)
    dg = dg * lower_bound * sig * (1 - sig) * av
    bs = beta.sigmoid()
    db = db * bs * (1 - bs)
    return (norm(q, dq), norm(k, dk), dv, db, dg, (dg * x).sum((0, 1, 3)), dg.sum((0, 1)))


def mm(a, b):
    return a.bfloat16().float() @ b.bfloat16().float()


def stable(s, do, du, dh, scale, residual=False, fp32=False):
    matmul = (lambda a, b: torch.matmul(a.float(), b.float())) if fp32 else mm
    q, k, v, g, beta, akk_matrix = [s[n].float() for n in ("q_c", "k_c", "v_c", "gk_c", "beta_c", "akk")]
    b, nc, n, c, d = q.shape
    h_state = s["h"][:, :-1].transpose(-1, -2).float()
    dh_state = dh[:, 1:].transpose(-1, -2).float()
    v_new_chunk = s["v_new"].transpose(-1, -2).float()
    do_chunk = do.reshape(b, nc, c, n, d).transpose(2, 3).float()
    gate_exp = g.exp()
    end_gate_exp = (g[..., -1:, :] - g).exp()
    q_gated = q * gate_exp
    k_gated_beta = k * gate_exp * beta[..., None]
    dk_initial = matmul(v_new_chunk, dh_state.transpose(-1, -2)) * end_gate_exp
    dg = -dk_initial * k
    dg[..., -1, :] += g[..., -1, :].exp() * (h_state * dh_state).sum(-1) + (dk_initial * k).sum(-2)
    da = matmul(do_chunk, v_new_chunk.transpose(-1, -2)).tril() * scale
    dq = matmul(do_chunk, h_state.transpose(-1, -2)) * scale * gate_exp
    dg += dq * q
    u_chunk = du.float()
    dvb = matmul(akk_matrix.transpose(-1, -2), u_chunk)
    if not fp32 and residual:
        dvb += matmul(akk_matrix.transpose(-1, -2), u_chunk - u_chunk.bfloat16().float())
    dv = dvb * beta[..., None]
    db = (dvb * v).sum(-1)
    dw = -matmul(u_chunk, h_state.transpose(-1, -2))
    daa = matmul(u_chunk, (v * beta[..., None]).transpose(-1, -2)) + matmul(dw, k_gated_beta.transpose(-1, -2))
    dwk = matmul(akk_matrix.transpose(-1, -2), dw)
    dkw = dwk * gate_exp * beta[..., None]
    dk = dk_initial + dkw
    dg += dkw * k
    db += (dwk * k * gate_exp).sum(-1)
    dl = -matmul(akk_matrix.transpose(-1, -2), matmul(daa.tril(-1), akk_matrix.transpose(-1, -2))).tril(-1)

    def dot(a, b):
        z = matmul(a, b)
        if residual:
            z += matmul(a - a.bfloat16().float(), b) + matmul(a, b - b.bfloat16().float())
            if residual == 2:
                z += matmul(a - a.bfloat16().float(), b - b.bfloat16().float())
        return z

    for i in range(0, 64, 16):
        for j in range(0, i + 1, 16):
            ii = slice(i, i + 16)
            jj = slice(j, j + 16)
            ref = g[..., i + 8: i + 9, :] if i == j else g[..., i: i + 1, :]
            ei = (g[..., ii, :] - ref).exp()
            ej = (ref - g[..., jj, :]).exp()
            dij = da[..., ii, jj].clone()
            if i == j:
                diag = dij.diagonal(dim1=-2, dim2=-1).clone()
                dij.diagonal(dim1=-2, dim2=-1).zero_()
                dq[..., ii, :] += diag[..., None] * k[..., ii, :]
                dk[..., ii, :] += diag[..., None] * q[..., ii, :]
            qa = dot(dij, k[..., jj, :] * ej) * ei
            kr = dot(dl[..., ii, jj], k[..., jj, :] * ej) * ei
            kt = (
                dot(dij.transpose(-1, -2), q[..., ii, :] * ei)
                + dot(dl[..., ii, jj].transpose(-1, -2), k[..., ii, :] * beta[..., ii, None] * ei)
            ) * ej
            dq[..., ii, :] += qa
            dk[..., ii, :] += kr * beta[..., ii, None]
            dk[..., jj, :] += kt
            db[..., ii] += (kr * k[..., ii, :]).sum(-1)
            dg[..., ii, :] += qa * q[..., ii, :] + kr * beta[..., ii, None] * k[..., ii, :]
            dg[..., jj, :] -= kt * k[..., jj, :]
    dg = dg.flip(-2).cumsum(-2).flip(-2)

    def unpack(x):
        return x.transpose(2, 3).reshape(b, nc * 64, n, *x.shape[4:]).float()

    return tuple(unpack(x) for x in (dq, dk, dv, db, dg))
