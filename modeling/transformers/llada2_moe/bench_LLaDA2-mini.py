# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""LLaDA2.0-mini E2E benchmark: warmup + N measurement iterations.

LLaDA2 uses block-wise masked diffusion (not autoregressive).
Warmup iterations absorb JIT compilation; measurement iterations are timed separately.

Usage:
    python modeling/transformers/llada2_moe/bench_LLaDA2-mini.py \
        --model-path /path/to/LLaDA2.0-mini [--use_pypto] [--device 14]
"""

import argparse
import atexit
import json
import logging
import os
import shutil
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

# Patch: add 'default' rope type if missing (transformers >= 4.57 compat)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                     .to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser(description="LLaDA2.0-mini E2E benchmark")
parser.add_argument("--model-path", required=True, help="Path to model weights")
parser.add_argument("--device", default=14, type=int, help="NPU device ID")
parser.add_argument("--prompt", default="Explain the concept of mixture of experts in neural networks.",
                    help="Input prompt")
parser.add_argument("--output_length", default=100, type=int, help="Generation length (tokens)")
parser.add_argument("--steps", default=32, type=int, help="Diffusion steps per block")
parser.add_argument("--block_length", default=32, type=int, help="Block length for masked diffusion")
parser.add_argument("--warmup", default=3, type=int, help="Warmup iterations (absorbs JIT)")
parser.add_argument("--iters", default=10, type=int, help="Measurement iterations")
parser.add_argument("--use_pypto", action="store_true", help="Enable PyPTO fused kernels")
parser.add_argument("--forward-only", action="store_true", help="Benchmark model.forward instead of generate")
parser.add_argument("--graph", action="store_true",
                    help="NPUGraph-capture arm: single fixed block, static routing, captured forward "
                         "replayed per denoising step. With --use_pypto -> grouped_gemm; else dense loop.")
parser.add_argument("--report-file", default=None, help="Output JSON report path")
args = parser.parse_args()


def setup_device(device):
    torch.npu.set_device(device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)


def setup_pypto(model_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    patched_src = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                               "llada2_moe", "modeling_llada2_moe.py")
    patched_dst = os.path.join(model_path, "modeling_llada2_moe.py")
    if os.path.exists(patched_src):
        backup = patched_dst + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(patched_dst, backup)
        shutil.copy2(patched_src, patched_dst)
        atexit.register(lambda: shutil.copy2(backup, patched_dst))
        logging.info(f"[PyPTO] Installed patched modeling to {patched_dst}")
    cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    shutil.rmtree(os.path.join(cache_root, "LLaDA2_dot_0_hyphen_mini"), ignore_errors=True)
    from ops.pypto_tensor import llada2_moe as llada2_kernels
    llada2_kernels.USE_PTO_EXPERT_FFN = True
    sys.modules["llada2_pto_kernels"] = llada2_kernels
    logging.info("[PyPTO] Expert FFN kernel enabled")


def load_model_and_tokenizer(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.info(f"Loading model from {model_path} ...")
    torch.npu.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map={"": f"npu:{device}"},
        local_files_only=True, trust_remote_code=True,
    )
    torch.npu.synchronize()
    model_load_s = time.perf_counter() - t0
    peak_load_mem = torch.npu.max_memory_allocated() / 1024**2
    logging.info(f"Model loaded in {model_load_s:.1f}s (peak {peak_load_mem:.0f} MB)")
    torch.npu.reset_peak_memory_stats()
    return tokenizer, model, model_load_s, peak_load_mem


def prepare_inputs(tokenizer, prompt, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(f"npu:{device}")
    attention_mask = torch.ones(
        (input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1]),
        dtype=torch.bool,
        device=input_ids.device,
    )
    return input_ids, attention_mask


def timed_call(fn):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = fn()
    torch.npu.synchronize()
    return outputs, time.perf_counter() - t0


def run_forward_benchmark(model, input_ids, attention_mask):
    logging.info(f"\nWarmup forward ({args.warmup} iterations)...")
    for idx in range(args.warmup):
        _, elapsed = timed_call(lambda: model(input_ids=input_ids, attention_mask=attention_mask))
        logging.info(f"  warmup {idx + 1}: {elapsed:.3f}s")

    logging.info(f"\nMeasurement forward ({args.iters} iterations)...")
    times, logits_shape, argmax_token = [], None, None
    for idx in range(args.iters):
        outputs, elapsed = timed_call(lambda: model(input_ids=input_ids, attention_mask=attention_mask))
        logits = outputs.logits
        logits_shape = list(logits.shape)
        argmax_token = int(logits[:, -1, :].argmax(dim=-1)[0].item())
        times.append(elapsed)
        logging.info(f"  iter {idx + 1:2d}: {elapsed * 1000:.3f} ms, logits={logits_shape}, argmax={argmax_token}")
    return times, logits_shape, argmax_token


def build_forward_result(context, times, logits_shape, argmax_token):
    peak_mem = torch.npu.max_memory_allocated() / 1024**2
    result = dict(context)
    result.update({
        "benchmark": "forward",
        "forward_peak_mem_mb": round(peak_mem, 1),
        "time_mean_ms": round(statistics.mean(times) * 1000, 3),
        "time_std_ms": round(statistics.stdev(times) * 1000, 3) if len(times) > 1 else 0,
        "time_min_ms": round(min(times) * 1000, 3),
        "time_max_ms": round(max(times) * 1000, 3),
        "logits_shape": logits_shape,
        "argmax_token": argmax_token,
        "per_iter": [{"time_ms": round(t * 1000, 3)} for t in times],
    })
    return result


def run_generate_benchmark(model, input_ids, input_len):
    gen_kwargs = dict(gen_length=args.output_length, steps=args.steps,
                      block_length=args.block_length, temperature=0.0)
    logging.info(f"\nWarmup ({args.warmup} iterations)...")
    for idx in range(args.warmup):
        _, elapsed = timed_call(lambda: model.generate(input_ids, **gen_kwargs))
        logging.info(f"  warmup {idx + 1}: {elapsed:.2f}s")

    logging.info(f"\nMeasurement ({args.iters} iterations)...")
    times, tokens_list = [], []
    for idx in range(args.iters):
        outputs, elapsed = timed_call(lambda: model.generate(input_ids, **gen_kwargs))
        n_tokens = outputs.shape[1] - input_len
        tps = n_tokens / elapsed if elapsed > 0 else 0
        times.append(elapsed)
        tokens_list.append(n_tokens)
        logging.info(f"  iter {idx + 1:2d}: {elapsed:.3f}s, {n_tokens} tokens, {tps:.1f} tok/s")
    return times, tokens_list


def build_generate_result(context, times, tokens_list):
    peak_mem = torch.npu.max_memory_allocated() / 1024**2
    tps_list = [t / e for t, e in zip(tokens_list, times)]
    result = dict(context)
    result.update({
        "output_length": args.output_length,
        "generate_peak_mem_mb": round(peak_mem, 1),
        "time_mean_s": round(statistics.mean(times), 3),
        "time_std_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
        "time_min_s": round(min(times), 3),
        "time_max_s": round(max(times), 3),
        "tps_mean": round(statistics.mean(tps_list), 1),
        "tps_std": round(statistics.stdev(tps_list), 1) if len(tps_list) > 1 else 0,
        "tps_min": round(min(tps_list), 1),
        "tps_max": round(max(tps_list), 1),
        "per_iter": [{"time_s": round(t, 3), "tokens": n, "tps": round(n / t, 1)}
                     for t, n in zip(times, tokens_list)],
    })
    return result


def print_summary(mode, result):
    if result.get("benchmark") == "forward":
        logging.info(f"\n--- {mode.upper()} FORWARD Summary ---")
        logging.info(f"  Time:     {result['time_mean_ms']:.3f} +/- {result['time_std_ms']:.3f} ms "
              f"(min={result['time_min_ms']:.3f}, max={result['time_max_ms']:.3f})")
        logging.info(f"  Peak mem: {result['forward_peak_mem_mb']:.0f} MB")
        logging.info(f"  Logits:   {result['logits_shape']}, argmax={result['argmax_token']}")
        return

    logging.info(f"\n--- {mode.upper()} Summary ---")
    logging.info(f"  Time:       {result['time_mean_s']:.3f} +/- {result['time_std_s']:.3f}s "
          f"(min={result['time_min_s']:.3f}, max={result['time_max_s']:.3f})")
    logging.info(f"  Throughput: {result['tps_mean']:.1f} +/- {result['tps_std']:.1f} tok/s "
          f"(min={result['tps_min']:.1f}, max={result['tps_max']:.1f})")
    logging.info(f"  Peak mem:   {result['generate_peak_mem_mb']:.0f} MB")


def run_benchmark():
    mode = "pypto" if args.use_pypto else "baseline"
    logging.info(f"\n{'='*70}")
    logging.info(f"  LLaDA2.0-mini E2E Benchmark — {mode.upper()}")
    logging.info(f"{'='*70}")

    setup_device(args.device)
    if args.use_pypto:
        setup_pypto(args.model_path)

    tokenizer, model, model_load_s, peak_load_mem = load_model_and_tokenizer(args.model_path, args.device)
    input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt, args.device)
    input_len = input_ids.shape[1]
    logging.info(f"Input tokens: {input_len}, Output length (gen_length): {args.output_length}")

    context = {
        "model": "LLaDA2.0-mini",
        "mode": mode,
        "device": f"npu:{args.device}",
        "input_tokens": input_len,
        "model_load_s": round(model_load_s, 2),
        "model_load_peak_mem_mb": round(peak_load_mem, 1),
        "warmup_iters": args.warmup,
        "num_iters": args.iters,
    }
    if args.forward_only:
        times, logits_shape, argmax_token = run_forward_benchmark(model, input_ids, attention_mask)
        result = build_forward_result(context, times, logits_shape, argmax_token)
    else:
        times, tokens_list = run_generate_benchmark(model, input_ids, input_len)
        result = build_generate_result(context, times, tokens_list)
    print_summary(mode, result)
    return result


def _setup_pypto_kernels():
    """Inject the LLaDA2 PyPTO kernels when --use_pypto; return grouped_gemm (or None)."""
    if not args.use_pypto:
        return None
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    from ops.pypto_tensor import llada2_moe as llada2_kernels
    llada2_kernels.USE_PTO_EXPERT_FFN = True
    sys.modules["llada2_pto_kernels"] = llada2_kernels
    return llada2_kernels.grouped_gemm


def _install_capture_patches(model, window, device):
    """Neutralize the mask-prep + rotary host syncs that abort capture; return (cos, sin, pos_ids)."""
    pos_ids = torch.arange(window, device=device).unsqueeze(0)
    modeling = sys.modules[model.__class__.__module__]
    if hasattr(modeling, "_prepare_4d_causal_attention_mask_for_sdpa"):
        def passthrough_mask(attention_mask, *fargs, **fkwargs):
            return attention_mask
        setattr(modeling, "_prepare_4d_causal_attention_mask_for_sdpa", passthrough_mask)
    rotary = next((m for m in model.modules()
                   if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn")), None)
    with torch.no_grad():
        cos, sin = rotary.forward(torch.zeros(1, window, 1, device=device, dtype=torch.bfloat16), pos_ids)
    cos, sin = cos.contiguous(), sin.contiguous()

    def cached_rotary(hidden_states, position_ids=None):
        return cos, sin
    rotary.forward = cached_rotary
    return cos, sin, pos_ids


def _capture_forward(forward_fn):
    """Warm up on a side stream, then NPUGraph-capture forward_fn; return (graph, captured_output)."""
    torch.npu.synchronize()
    side_stream = torch.npu.Stream()
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        for _ in range(3):
            forward_fn()
    torch.npu.current_stream().wait_stream(side_stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = forward_fn()
    return graph, output


def _make_static_moe_forward(use_pypto, grouped_gemm):
    """Build a graph-capturable static-route MoE forward (every token -> the first top_k experts)."""
    import torch.nn.functional as functional

    def static_forward(self, hidden_states):
        getattr(self, "_ensure_pypto_weights")()
        identity = hidden_states
        bsz, seq, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        num_tokens = flat.shape[0]
        w13 = getattr(self, "_pypto_w13_stack")
        w2 = getattr(self, "_pypto_w2_stack")
        num_experts = w13.shape[0]
        inter = w2.shape[1]
        width = self.num_experts_per_tok
        if use_pypto:
            sorted_x = flat.repeat(width, 1).contiguous()
            counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
            counts[:width] = num_tokens
            cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
            cumsum[1:] = torch.cumsum(counts, 0).to(torch.int32)
            out = torch.empty_like(sorted_x)
            grouped_gemm(sorted_x, w13.reshape(num_experts * hidden, 2 * inter).contiguous(),
                         w2.reshape(num_experts * inter, hidden).contiguous(), cumsum, out,
                         num_experts=num_experts, hidden_size=hidden, intermediate_size=inter)
            combined = out.view(width, num_tokens, hidden).sum(0).mul_(1.0 / width)
        else:
            combined = torch.zeros(num_tokens, hidden, dtype=flat.dtype, device=flat.device)
            for expert in range(width):
                gate_up = torch.matmul(flat, w13[expert])
                combined = combined + torch.matmul(
                    functional.silu(gate_up[:, :inter]) * gate_up[:, inter:], w2[expert])
            combined = combined.mul_(1.0 / width)
        combined = combined.view(bsz, seq, hidden)
        if getattr(self.config, "num_shared_experts", None) is not None:
            combined = combined + self.shared_experts(identity)
        return combined, (None, None)

    return static_forward


def _install_static_moe(model, use_pypto, grouped_gemm):
    """Replace every MoE block's forward with the static-route capturable forward."""
    import types
    static_forward = _make_static_moe_forward(use_pypto, grouped_gemm)
    for block in model.modules():
        if all(hasattr(block, a) for a in ("_ensure_pypto_weights", "num_experts_per_tok", "experts")):
            block.forward = types.MethodType(static_forward, block)


def _measure_and_report(run_once, mode, device, dims):
    """Warm up, time iters, build the LLaDA2 result dict, and print the summary."""
    for _ in range(dims["warmup"]):
        run_once()
    times = [run_once() for _ in range(dims["iters"])]
    best = min(times)
    gen_len = dims["gen_len"]
    tps_list = [gen_len / t for t in times]                  # per-iter throughput, then AVERAGE
    tps_mean = round(statistics.mean(tps_list), 1)
    tps_std = round(statistics.stdev(tps_list), 1) if len(tps_list) > 1 else 0.0
    result = {
        "model": "LLaDA2.0-mini", "mode": mode, "device": device,
        "block_length": dims["window"], "prompt_in_block": dims["prompt_len"],
        "gen_per_block": gen_len, "steps": dims["num_steps"],
        "tps_mean": tps_mean, "tps_std": tps_std, "tps_max": round(max(tps_list), 1),
        "time_mean_s": round(statistics.mean(times), 3), "time_min_s": round(best, 3),
        "warmup_iters": dims["warmup"], "num_iters": dims["iters"],
        "generate_peak_mem_mb": round(torch.npu.max_memory_allocated() / 1024 ** 2, 1),
    }
    logging.info(f"\n--- {mode.upper()} Summary ---")
    logging.info(f"  Throughput: {tps_mean:.1f} +/- {tps_std:.1f} tok/s "
          f"(avg of {dims['iters']}, mean {result['time_mean_s']:.3f}s/block)")
    return result


def run_graph_benchmark():
    """NPUGraph-capture arm: single fixed diffusion block, static routing, captured forward replayed
    per denoising step. Capture needs attn_implementation='eager' (SDPA fused kernel aborts capture,
    107025), a prebuilt 0-mask, and precomputed cos/sin - see _install_capture_patches.
    """
    use_pypto = args.use_pypto
    mode = "pypto+graph" if use_pypto else "graph"
    sep = "=" * 70
    logging.info(f"\n{sep}\n  LLaDA2.0-mini E2E Benchmark - {mode.upper()} (NPUGraph capture)\n{sep}")
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    os.environ["TILE_FWK_DEVICE_ID"] = str(args.device)
    grouped_gemm = _setup_pypto_kernels()

    # Both graph arms need the in-repo modeling (its MoE block exposes _ensure_pypto_weights /
    # _pypto_w13_stack that _install_static_moe routes through). Without it the stock remote
    # modeling's MoE forward runs, and its tokens_per_expert.cpu()/.item() host syncs abort capture
    # (107030). run_benchmark installs it via setup_pypto; do the same here for vec + pypto.
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _psrc = os.path.join(_repo, "src", "pypto_gym", "transformers", "llada2_moe", "modeling_llada2_moe.py")
    _pdst = os.path.join(args.model_path, "modeling_llada2_moe.py")
    if os.path.exists(_psrc):
        _bak = _pdst + ".orig"
        if os.path.exists(_pdst) and not os.path.exists(_bak):
            shutil.copy2(_pdst, _bak)
        shutil.copy2(_psrc, _pdst)
        atexit.register(lambda: os.path.exists(_bak) and shutil.copy2(_bak, _pdst))
        logging.info(f"[graph] Installed in-repo modeling to {_pdst}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": device},
        local_files_only=True, trust_remote_code=True, attn_implementation="eager").eval()
    mask_id = 156895
    window = args.block_length
    prompt_len = max(1, min(args.output_length if args.output_length < window else window // 4, window - 1))
    gen_len = window - prompt_len
    num_steps = args.steps

    _install_static_moe(model, use_pypto, grouped_gemm)
    cur_x = torch.full((1, window), mask_id, dtype=torch.long, device=device)
    cur_x[:, :prompt_len] = torch.randint(0, 100000, (1, prompt_len), device=device)
    zmask = torch.zeros(1, 1, window, window, device=device, dtype=torch.bfloat16)
    cos, sin, pos_ids = _install_capture_patches(model, window, device)
    body = next(m for m in model.modules()
                if all(hasattr(m, a) for a in ("word_embeddings", "layers", "norm")))
    lm_head = model.get_output_embeddings()
    reveal_per_step = max(1, gen_len // num_steps)

    @torch.no_grad()
    def forward_fn():
        hidden = body.word_embeddings(cur_x)
        for layer in body.layers:
            hidden = layer(hidden, attention_mask=zmask, position_ids=pos_ids,
                           position_embeddings=(cos, sin), use_cache=False)[0]
        return lm_head(body.norm(hidden))

    def denoise_host(logits):
        next_ids = logits[0, prompt_len:window].argmax(-1)
        masked = (cur_x[0, prompt_len:window] == mask_id).nonzero(as_tuple=True)[0][:reveal_per_step]
        if masked.numel() > 0:
            cur_x[0, prompt_len + masked] = next_ids[masked]

    def run_once():
        cur_x[:, prompt_len:].fill_(mask_id)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            graph.replay()
            denoise_host(graph_logits)
        torch.npu.synchronize()
        return time.perf_counter() - start

    graph, graph_logits = _capture_forward(forward_fn)
    logging.info("[graph] capture OK")
    dims = {"window": window, "prompt_len": prompt_len, "gen_len": gen_len, "num_steps": num_steps,
            "warmup": args.warmup, "iters": args.iters}
    return _measure_and_report(run_once, mode, device, dims)


if __name__ == "__main__":
    result = run_graph_benchmark() if args.graph else run_benchmark()

    if args.report_file:
        report_path = args.report_file
    else:
        tag = ("pypto_graph" if args.use_pypto else "graph") if args.graph \
            else ("pypto" if args.use_pypto else "baseline")
        report_path = os.path.join(os.path.dirname(__file__), f"bench_{tag}.json")

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"\nReport saved: {report_path}")
