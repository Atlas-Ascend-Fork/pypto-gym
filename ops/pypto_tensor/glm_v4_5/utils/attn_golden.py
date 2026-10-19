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
import math
import torch
import torch_npu


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=actual_seq_len.device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, actual_seqs):
    b = actual_seqs.shape[0]
    n2 = kr_cache_out.shape[2]
    d = kr_cache_out.shape[3]
    block_size = kr_cache_out.shape[1]
    dtype = kv_cache_out.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    kv_max = (torch.max(actual_seqs).item() + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    k_cache = torch.zeros([b, kv_max, n2, d], dtype=dtype).to(kr_cache_out.device)
    v_cache = torch.zeros([b, kv_max, n2, d], dtype=dtype).to(kr_cache_out.device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


def ifa_golden(q, k, v, block_table, actual_seqs, out, is_high_precision=True, is_use_page=False):
    if is_high_precision:
        fp64 = torch.float64
        q = q.to(fp64)
        k = k.to(fp64)
        v = v.to(fp64)
        b = actual_seqs.shape[0]
        bs = q.shape[0]
        s1 = bs // b
        nkv = k.shape[2]
        d = k.shape[3]
        softmax_scale = d ** -0.5
        k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, actual_seqs)

        for i in range(b):
            for j in range(s1):
                for n2_idx in range(nkv):
                    # 从 torch tensor 获取值
                    kv_seq_len = actual_seqs[i].item()  # 使用 .item() 获取标量值
                    seq_len = kv_seq_len - s1 + 1 + j
                    q_bs = q[i * s1 + j]
                    k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    # MM1: 矩阵乘法
                    qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))  # 1,nq, d  -> n_q,d @ d, s2_actual_len
                    qk_ele_res = qk_bmm_res * softmax_scale
                    # Softmax计算
                    softmax_res, _, _ = softmax(qk_ele_res, True)

                    # MM2: 矩阵乘法
                    bmm2_res = torch.matmul(softmax_res, v_bs)

                    # 存储结果
                    out[i * s1 + j] = bmm2_res
    else:
        ifa_flash_torch(q=q, k=k, v=v, block_table=block_table, kv_act_seqs=actual_seqs, out=out)


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32))


def ifa_flash_torch(q, k, v, block_table, kv_act_seqs, out, is_fp32=False):
    fp32 = torch.float32
    if is_fp32:
        q = q.to(fp32)
        k = k.to(fp32)
        v = v.to(fp32)

    # ========== 1. 提取维度信息（与原代码一致） ==========
    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b  # 每个样本的query序列长度
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape  # 补充block_num维度（原代码遗漏）
    g = n1 // n2  # 头数比例（n1必须是n2的整数倍）
    g_tile = g  # 与g一致，保留原变量名
    k_2d = k.reshape(-1, d)  # shape: [block_num*block_size*n2, d]
    v_2d = v.reshape(-1, d)  # shape: [block_num*block_size*n2, d]
    # q的重塑：将[b*s1, n1, d]重塑为[b*s1*n1, d]（与原代码一致）
    q_2d = q.reshape(-1, d)  # shape: [bs1*n1, d]

    s2_tile = block_size * 4

    # ========== 3. 循环处理每个样本、每个位置（保留原代码的循环逻辑） ==========
    # 遍历batch
    for b_idx in range(b):
        # 遍历每个query的位置
        for s1_idx in range(s1):
            # 计算当前kv的有效序列长度（原代码逻辑）
            cur_seq = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
            cur_seq = max(cur_seq.item(), 0)  # 防止负数（PyTorch标量需用.item()取数值）
            s2_loop = math.ceil(cur_seq / s2_tile)
            block_num = s2_tile // block_size

            # 遍历每个key/value头
            for n2_idx in range(n2):
                # 遍历头数比例g
                for g_idx in range(g // g_tile):
                    # ========== 4. 初始化中间变量（修正原代码的初始化错误） ==========
                    device = q.device
                    dtype = q.dtype
                    oi_upd = torch.zeros((g_tile, d), device=device, dtype=fp32)
                    li_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                    mi_upd = torch.zeros(g_tile, device=device, dtype=fp32)

                    for s2_idx in range(s2_loop):
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1 + s1_idx
                        n2g_ofs = n2_idx * g + g_idx * g_tile
                        actual_s2_tile = min(s2_tile, cur_seq - s2_idx * s2_tile)
                        actual_block_num = math.ceil(actual_s2_tile / block_size)

                        # 提取qi: shape [g_tile, d]
                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi_end = qi_start + g_tile
                        qi = q_2d[qi_start:qi_end, :]

                        # 组装kj_assemble: 从actual_block_num个block拼出 [s2_tile, d]
                        kj_assemble = torch.zeros((s2_tile, d), device=device, dtype=dtype)
                        for i in range(actual_block_num):
                            block_idx = block_table[b_idx][idx + i].item()
                            block_idx = max(block_idx, 0)
                            kj_assemble[i * block_size:(i + 1) * block_size, :] = \
                                k_2d[block_idx * block_size:(block_idx + 1) * block_size, :]

                        # ========== 6. 注意力计算 ==========
                        mm1 = matmul_proxy(qi, kj_assemble[:actual_s2_tile, :].t()).to(fp32)
                        muls_res = mm1 * (d ** -0.5)
                        tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)

                        # ========== 7. 累积更新oi、li、mi ==========
                        if s2_idx == 0:
                            tsub = muls_res - tilda_mij
                            tilda_pij = torch.exp(tsub)
                            tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                            # 组装vj_assemble
                            vj_assemble = torch.zeros((s2_tile, d), device=device, dtype=dtype)
                            for i in range(actual_block_num):
                                block_idx = block_table[b_idx][idx + i].item()
                                block_idx = max(block_idx, 0)
                                vj_assemble[i * block_size:(i + 1) * block_size, :] = \
                                    v_2d[block_idx * block_size:(block_idx + 1) * block_size, :]
                            oi_tmp = matmul_proxy(tilda_pij.to(dtype), vj_assemble[:actual_s2_tile, :]).to(fp32)
                            oi_upd = oi_tmp
                            li_upd = tilda_lij.squeeze(-1)
                            mi_upd = tilda_mij.squeeze(-1)
                        else:
                            mi = mi_upd.unsqueeze(-1)
                            max_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True)
                            tsub = muls_res - max_new
                            tilda_pij = torch.exp(tsub)
                            tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)

                            tsub2 = torch.sub(mi, max_new)
                            mi_upd = max_new.squeeze(-1)
                            update_mul = torch.exp(tsub2)
                            li = li_upd.unsqueeze(-1)
                            sum_new = li * update_mul + tilda_lij
                            li_upd = sum_new.squeeze(-1)
                            # 组装vj_assemble
                            vj_assemble = torch.zeros((s2_tile, d), device=device, dtype=dtype)
                            for i in range(actual_block_num):
                                block_idx = block_table[b_idx][idx + i].item()
                                block_idx = max(block_idx, 0)
                                vj_assemble[i * block_size:(i + 1) * block_size, :] = \
                                    v_2d[block_idx * block_size:(block_idx + 1) * block_size, :]
                            q1 = matmul_proxy(tilda_pij.to(dtype), vj_assemble[:actual_s2_tile, :]).to(fp32)
                            oi_upd = oi_upd * update_mul + q1

                        if s2_idx == s2_loop - 1:
                            li = li_upd.unsqueeze(-1)
                            oi_final = oi_upd / li
                            oi_upd_3d = oi_final.unsqueeze(0)
                            attn_out_start_col = n2g_ofs
                            attn_out_end_col = n2g_ofs + g_tile
                            if attn_out_end_col > out.shape[1]:
                                attn_out_end_col = out.shape[1]
                                attn_out_start_col = attn_out_end_col - g_tile
                            out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = oi_upd_3d.to(dtype)
    return out


def add_rms_norm_npu_golden(residual_input, x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    residual_input_fp32 = residual_input.to(torch.float32)
    x_fp32 = x_fp32 + residual_input_fp32
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16), x_fp32.to(torch.bfloat16)


def rms_norm_npu_golden(x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16)


def _apply_rotary_emb_neuron(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin):
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed


def attention_pre_golden(
        hidden_states,
        residual,
        input_layernorm_weight,
        input_layernorm_bias,
        qkv_proj_scale,
        qkv_proj_offset,
        qkv_proj_weight,
        qkv_proj_quant_bias,
        qkv_proj_deq_scale,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        eps
):
    bs = hidden_states.shape[0]
    d = q_norm_weight.shape[0]
    rotary_dim = d // 2
    q_size = qkv_proj_weight.shape[1] - 2 * d
    x_g, residual_g = add_rms_norm_npu_golden(hidden_states, residual, input_layernorm_weight, \
                                              input_layernorm_bias, eps)

    # matmul
    x_quant = torch_npu.npu_quantize(x_g, qkv_proj_scale, qkv_proj_offset, torch.qint8, -1, False)
    mm_golden = torch_npu.npu_quant_matmul(x_quant, qkv_proj_weight, qkv_proj_deq_scale, \
                                           bias=qkv_proj_quant_bias, output_dtype=torch.bfloat16)

    # split
    q_g, k_g, v_g = mm_golden.split([q_size, d, d], dim=-1)
    # nms norm
    q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // d, d)
    q_by_head = rms_norm_npu_golden(q_by_head, q_norm_weight, q_norm_bias, eps)

    k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // d, d)
    k_by_head = rms_norm_npu_golden(k_by_head, k_norm_weight, k_norm_bias, eps)

    # apply rope
    q_rot = q_by_head[..., :rotary_dim]
    q_pass = q_by_head[..., rotary_dim:]
    k_rot = k_by_head[..., :rotary_dim]
    k_pass = k_by_head[..., rotary_dim:]
    q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
    q_cat = torch.cat((q_r, q_pass), dim=-1)
    k_cat = torch.cat((k_r, k_pass), dim=-1)
    # post process
    q_r = q_cat.view(bs, q_size)
    k_r = k_cat.view(bs, d)
    return q_r, k_r, v_g, residual_g


def _scatter_update(value, cache, slots):
    bs = value.shape[0]
    block_number, block_size, n2, head_size = cache.shape
    for bs_idx in range(bs):
        index = slots[bs_idx]
        dim_0 = index // block_size
        dim_1 = index % block_size
        cache[dim_0, dim_1, :] = value[bs_idx][:]


def scatter_golden(k_r, v_g, key_cache, value_cache, slot_mapping):
    n2, d = key_cache.shape[-2], key_cache.shape[-1]
    b = k_r.shape[0]  # 这里严格意义上是bs，而不是b， 但当前s1为1
    key_scatter = k_r.view(b, n2, d)
    value_scatter = v_g.view(b, n2, d)
    _scatter_update(key_scatter, key_cache, slot_mapping)
    _scatter_update(value_scatter, value_cache, slot_mapping)


def attention_golden(
        hidden_states,
        residual,
        input_layernorm_weight,
        input_layernorm_bias,
        qkv_proj_scale,
        qkv_proj_offset,
        qkv_proj_weight,
        qkv_proj_quant_bias,
        qkv_proj_deq_scale,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        key_cache,
        value_cache,
        block_tables,
        actual_seq_lens,
        slot_mapping,
        eps,
        enable_residual,
        num_decode_tokens
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_r, k_r, v_g, residual = attention_pre_golden(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=input_layernorm_weight,
        input_layernorm_bias=input_layernorm_bias,
        qkv_proj_scale=qkv_proj_scale,
        qkv_proj_offset=qkv_proj_offset,
        qkv_proj_weight=qkv_proj_weight,
        qkv_proj_quant_bias=qkv_proj_quant_bias,
        qkv_proj_deq_scale=qkv_proj_deq_scale,
        q_norm_weight=q_norm_weight,
        q_norm_bias=q_norm_bias,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
        cos=cos,
        sin=sin,
        eps=eps
    )
    scatter_golden(k_r, v_g, key_cache, value_cache, slot_mapping)
    bs = q_r.shape[0]
    d = key_cache.shape[-1]
    q_r = q_r.view(bs, -1, d)
    attn_out = torch.zeros(q_r.shape, dtype=q_r.dtype).to(device=q_r.device)
    ifa_golden(q_r, key_cache, value_cache, block_tables, actual_seq_lens, attn_out, is_high_precision=False)
    return attn_out, residual
