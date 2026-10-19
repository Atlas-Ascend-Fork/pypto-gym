#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MiniMax (M2.7 / M3) single-die **E2E generation** benchmark (Ascend 910B).

One script, two variants selected by ``--variant {m27,m3}``. A *token-generating* benchmark, not a
repeated-forward microbench: it runs the real ``model.generate()`` autoregressive loop, **counts the
tokens it produces**, and reports the **average** throughput over ``--iters`` generations
(mean +/- std, not best-of), split into three regimes the way a serving benchmark decomposes a request:

  * **prefill** -- ingest the prompt and emit the first token (TTFT), via ``generate(max_new_tokens=1)``;
                   tok/s = prompt_len / prefill_time.
  * **decode**  -- the steady-state per-token loop, ``total - prefill`` over the remaining ``N-1``
                   tokens; tok/s = (N-1) / decode_time.
  * **both**    -- the full end-to-end generation; tok/s = N / total_time.

Arms (one process = one arm; ``bench_minimax.sh`` runs them and tabulates the JSON reports):

  * ``eager``    -- stock per-expert MoE FFN loop (M3 only; M2.7 always streams the fused kernel).
  * ``pypto``    -- routed expert FFN on the PyPTO fused grouped-GEMM kernel (streaming FP8 experts).
  * ``--graph``  -- NPUGraph-capture a fixed window, replay per generated token (decode-generate tok/s):
                    ``graph`` (NPU-friendly vectorized FFN) vs ``pypto+graph`` (fused grouped GEMM).

The static MoE route sends every token to the first ``top_k`` experts, so only those are dequantized
(FP8->BF16) onto the die once and captured. The fused-kernel *kernel-level* win (Eager vs PyPTO) is
measured separately by an operator microbench kept local-only (WARMUP=5 / ITERS=20); kernel
correctness is covered in-repo by ``tests/ops/minimax_m27/`` and ``tests/ops/minimax_m3/``.

The two variants differ only in: which model package they import, the expert activation passed to the
fused kernel (``silu`` for M2.7, ``swigluoai`` for M3), the model load entry point, and the
NPUGraph static-route forward. Everything else (arg parsing, env setup, warmup, input gen, capture,
generate loop, timing/report) is a single shared code path.

    MODEL_PATH=/path/to/MiniMax-M2.7 python bench_minimax.py --variant m27 --max-layers 2
    MODEL_PATH=/path/to/MiniMax-M3   python bench_minimax.py --variant m3 --max-layers 6 --use_pypto
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import namedtuple

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax.bench")

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu required (Ascend NPU).") from exc

from _variant import load_variant_model  # noqa: E402  (shared per-variant model load)


# --------------------------------------------------------------------------------------------------
# Per-variant constants -- the ONLY places the two models genuinely differ.
# --------------------------------------------------------------------------------------------------
_VARIANTS = {
    "m27": {
        "display": "MiniMax-M2.7",
        "activation": "silu",       # passed to the fused grouped_gemm kernel
        "vec_tile": "128",          # UB-fitting vector tile (910B)
    },
    "m3": {
        "display": "MiniMax-M3",
        "activation": "swigluoai",
        "vec_tile": "256",          # M3-tuned default (H=6144); still UB-fitting on 910B
    },
}


def parse_args():
    ap = argparse.ArgumentParser(description="MiniMax (M2.7 / M3) single-die E2E generation benchmark")
    ap.add_argument("--variant", choices=("m27", "m3"), required=True,
                    help="which MiniMax text backbone to benchmark")
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--device", type=int, default=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    ap.add_argument("--prompt", default="Explain mixture-of-experts routing in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="tokens to generate (N)")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="load only the first N decoder layers (fits-on-die; M3: use >=4 for MoE)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup generations (absorb JIT compile)")
    ap.add_argument("--iters", type=int, default=5, help="measured generations (averaged)")
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                    help="keep routed experts FP8 on host, dequant per layer (required for FP8 ckpt)")
    ap.add_argument("--use_pypto", action="store_true",
                    help="route MoE FFN through the PyPTO kernel (graph arm: fused vs vectorized)")
    ap.add_argument("--graph", action="store_true",
                    help="NPUGraph-capture a fixed window, replay per generated token (static MoE route)")
    ap.add_argument("--graph-window", type=int, default=32, help="fixed captured window W for --graph")
    ap.add_argument("--report-file", default=None, help="output JSON report path")
    # M2.7 8-die pipeline-parallel flags -- accepted for CLI/shell back-compat; guarded (no-op) for m3.
    ap.add_argument("--moe-impl", choices=("eager", "pypto"), default=None,
                    help="(m27 8-die) MoE implementation; alias for --use_pypto")
    ap.add_argument("--seq", type=int, default=None, help="(m27 8-die) prefill sequence length")
    ap.add_argument("--route", default=None, help="(m27 8-die) routing strategy")
    ap.add_argument("--active", type=int, default=None, help="(m27 8-die) active experts override")
    ap.add_argument("--prompt-text", default=None,
                    help="(m27 8-die) real text input (overrides --prompt)")
    args = ap.parse_args()
    # Back-compat normalization: let the legacy m27 shell flags drive the shared switches.
    if args.moe_impl is not None:
        args.use_pypto = args.moe_impl == "pypto"
    if args.prompt_text:
        args.prompt = args.prompt_text
    return args


def _mean_std(xs):
    return (round(statistics.mean(xs), 3),
            round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0)


def _build_inputs(args, dev):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
    return tok, ids


def _measure_generate(model, input_ids, args):
    """Warmup + timed generations: per iter, time prefill (TTFT, max_new_tokens=1) and the full
    generation, counting generated tokens. Returns (prompt_len, prefill_t, total_t, n_gen)."""
    prompt_len = input_ids.shape[1]
    num_new = args.max_new_tokens
    gen = dict(do_sample=False, use_cache=True)
    logger.info("  prompt_len=%d  max_new_tokens=%d  warmup=%d  iters=%d",
                prompt_len, num_new, args.warmup, args.iters)
    for i in range(args.warmup):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=num_new, **gen)
        torch.npu.synchronize()
        logger.info("  warmup %d: %.3fs", i + 1, time.perf_counter() - t0)
    prefill_t, total_t, n_gen = [], [], []
    for i in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=1, **gen)            # TTFT == prefill + 1 step
        torch.npu.synchronize()
        pf = time.perf_counter() - t0
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=num_new, **gen)
        torch.npu.synchronize()
        tot = time.perf_counter() - t0
        g = int(out.shape[1] - prompt_len)
        prefill_t.append(pf)
        total_t.append(tot)
        n_gen.append(g)
        logger.info("  iter %2d: prefill=%.3fs  total=%.3fs  gen=%d tok  "
                    "(decode %.1f tok/s, e2e %.1f tok/s)",
                    i + 1, pf, tot, g, (g - 1) / max(tot - pf, 1e-9), g / tot)
    return prompt_len, prefill_t, total_t, n_gen


def _summarize_generate(meta, timings):
    """Average prefill/decode/both throughput (mean +/- std, not best-of), build the result dict,
    and log the summary. ``timings`` = (prompt_len, num_new, prefill_t, total_t, n_gen, peak_mem)."""
    prompt_len, num_new, prefill_t, total_t, n_gen, peak_mem = timings
    pf_m, pf_s = _mean_std([prompt_len / p for p in prefill_t])
    dc_m, dc_s = _mean_std([(g - 1) / max(t - p, 1e-9) for g, t, p in zip(n_gen, total_t, prefill_t)])
    e2_m, e2_s = _mean_std([g / t for g, t in zip(n_gen, total_t)])
    result = {
        **meta, "prompt_len": prompt_len, "max_new_tokens": num_new, "gen_tokens": n_gen,
        "peak_mem_mb": round(peak_mem, 1), "prefill_time_s": _mean_std(prefill_t),
        "decode_time_s": _mean_std([t - p for t, p in zip(total_t, prefill_t)]),
        "total_time_s": _mean_std(total_t),
        "prefill_tps_mean": pf_m, "prefill_tps_std": pf_s,
        "decode_tps_mean": dc_m, "decode_tps_std": dc_s,
        "e2e_tps_mean": e2_m, "e2e_tps_std": e2_s,
    }
    logger.info("\n--- %s Summary (avg of %d gens) ---", meta["mode"].upper(), meta["iters"])
    logger.info("  prefill: %.1f +/- %.1f tok/s   (%.3fs to first token)",
                pf_m, pf_s, statistics.mean(prefill_t))
    logger.info("  decode : %.1f +/- %.1f tok/s", dc_m, dc_s)
    logger.info("  e2e    : %.1f +/- %.1f tok/s", e2_m, e2_s)
    logger.info("  peak mem: %.0f MB", peak_mem)
    return result


def run_generate(args, dev):
    """eager / pypto E2E generation arm: real generate(), counts tokens, averages prefill/decode/both."""
    spec = _VARIANTS[args.variant]
    # M2.7 always streams the fused kernel; M3 has an eager arm selectable via --use_pypto.
    mode = "pypto" if (args.variant == "m27" or args.use_pypto) else "eager"
    logger.info("\n%s\n  %s E2E generation -- %s\n%s",
                "=" * 70, spec["display"], mode.upper(), "=" * 70)

    torch.npu.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, cfg, n_patched = load_variant_model(args, dev, streaming=True)
    load_s = time.perf_counter() - t0
    logger.info("  loaded %d layers in %.1fs (patched %d MoE block(s) with PyPTO)",
                cfg.num_hidden_layers, load_s, n_patched)
    if args.variant == "m3" and args.use_pypto and n_patched == 0:
        logger.warning("  NOTE: 0 MoE blocks patched -- max_layers=%s has no MoE layer "
                       "(layers 0-%d are dense); use --max-layers >= %d for a real PyPTO comparison.",
                       args.max_layers, cfg.first_k_dense_replace - 1, cfg.first_k_dense_replace + 1)

    _tok, input_ids = _build_inputs(args, dev)
    prompt_len, prefill_t, total_t, n_gen = _measure_generate(model, input_ids, args)
    peak_mem = torch.npu.max_memory_allocated() / 1024 ** 2
    meta = {"model": spec["display"], "mode": mode, "device": dev, "layers": cfg.num_hidden_layers,
            "moe_patched": n_patched, "warmup": args.warmup, "iters": args.iters,
            "model_load_s": round(load_s, 2)}
    return _summarize_generate(meta, (prompt_len, args.max_new_tokens, prefill_t, total_t, n_gen, peak_mem))


# --------------------------------------------------------------------------------------------------
# NPUGraph static-route forward -- per-variant install (the only graph difference between M2.7/M3).
# --------------------------------------------------------------------------------------------------
def _install_static_moe_m27(body, dev, use_pypto):
    """M2.7: dequant the first top_k experts (FP8->BF16) to device + swap each MoE block to a
    capturable static-route forward (every token to experts 0..top_k-1)."""
    import types
    from torch.nn.functional import linear, silu
    from ops.pypto_tensor.minimax import MoeDims, grouped_gemm
    from pypto_gym.transformers.minimax_m27.modeling_minimax_m27 import _expert_weight

    def _dequant(block):
        width = block.top_k
        w1s, w3s, w2s = [], [], []
        for idx in range(width):
            exp = block.experts[idx]
            w1s.append(_expert_weight(exp.w1).to(dev))       # [inter, hidden]
            w3s.append(_expert_weight(exp.w3).to(dev))       # [inter, hidden]
            w2s.append(_expert_weight(exp.w2).to(dev))       # [hidden, inter]
        inter, hidden = w1s[0].shape
        if use_pypto:
            w13_flat = torch.empty(width * hidden, 2 * inter, dtype=torch.bfloat16, device=dev)
            w2_flat = torch.empty(width * inter, hidden, dtype=torch.bfloat16, device=dev)
            for idx in range(width):
                w13_flat[idx * hidden:(idx + 1) * hidden] = torch.cat([w1s[idx], w3s[idx]], 0).t().contiguous()
                w2_flat[idx * inter:(idx + 1) * inter] = w2s[idx].t().contiguous()
            block.static_weights = (w13_flat, w2_flat, inter, width)
        else:
            block.static_weights = (w1s, w3s, w2s, inter, width)

    def _make_forward(block):
        width = block.top_k

        def static_forward(self, hidden_states):
            bsz, seq, hidden = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden)
            n_tok = flat.shape[0]
            if use_pypto:
                w13_flat, w2_flat, inter, _ = self.static_weights
                sorted_x = flat.repeat(width, 1).contiguous()
                # Build expert_cumsum ONCE outside NPUGraph capture and reuse the persistent tensor.
                # The kernel declares expert_cumsum host-ready; rebuilding it (arange) inside the
                # captured region makes that host-read force a mid-capture device->host sync -> stale
                # replay schedule -> aicore 507011. The static route is fixed, so cache it once.
                cumsum = getattr(self, "cached_cumsum", None)
                if cumsum is None:
                    cumsum = torch.arange(0, width + 1, dtype=torch.int32, device=flat.device) * n_tok
                    self.cached_cumsum = cumsum
                out = torch.empty_like(sorted_x)
                grouped_gemm(sorted_x, (w13_flat, w2_flat), cumsum, out,
                             MoeDims(width, hidden, inter))
                routed = out.view(width, n_tok, hidden).sum(0).mul_(1.0 / width)
            else:
                w1s, w3s, w2s, _inter, _ = self.static_weights
                routed = torch.zeros(n_tok, hidden, dtype=flat.dtype, device=flat.device)
                for idx in range(width):
                    routed = routed + linear(silu(linear(flat, w1s[idx])) * linear(flat, w3s[idx]), w2s[idx])
                routed = routed.mul_(1.0 / width)
            return routed.view(bsz, seq, hidden), None
        return static_forward

    for layer in body.layers:
        block = layer.block_sparse_moe
        _dequant(block)
        block.forward = types.MethodType(_make_forward(block), block)


def _install_static_moe_m3(body, use_pypto):
    """M3: force dense GQA on every attention (bypass the MSA indexer) and swap each MoE block to the
    static-route forward, so the whole decoder becomes NPUGraph-capturable."""
    import types
    from pypto_gym.transformers.minimax_m3.modeling_minimax_m3 import _swiglu_oai

    def _make_forward(block):
        num_experts = block.experts.num_experts
        width = block.top_k
        scaling = block.routed_scaling_factor

        def static_forward(self, hidden_states):
            bsz, seq, hidden = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden)
            n_tok = flat.shape[0]
            if use_pypto:
                self.experts.ensure_pypto_weights()
                sorted_x = flat.repeat(width, 1).contiguous()
                counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device)
                counts[:width] = n_tok
                out = self.experts.grouped_ffn(sorted_x, counts)
                routed = out.view(width, n_tok, hidden).sum(0).mul_(1.0 / width)
            else:
                routed = torch.zeros(n_tok, hidden, dtype=flat.dtype, device=flat.device)
                for idx in range(width):
                    exp = self.experts[idx]
                    routed = routed + exp.w2(_swiglu_oai(exp.w1(flat), exp.w3(flat), exp.alpha, exp.limit))
                routed = routed.mul_(1.0 / width)
            routed = routed.view(bsz, seq, hidden) * scaling
            if self.shared_experts is not None:
                routed = routed + self.shared_experts(hidden_states)
            return routed, None
        return static_forward

    for layer in body.layers:
        layer.self_attn.is_sparse = False
        layer.self_attn.use_msa_sparse = False
        if getattr(layer, "is_moe", False):
            block = layer.block_sparse_moe
            block.forward = types.MethodType(_make_forward(block), block)
            if use_pypto:
                # static route -> cache expert_cumsum once (pre-capture) so the ready_on_host read
                # doesn't force a mid-capture device->host sync under NPUGraph (avoids 507011).
                block.experts.static_cumsum_cache = True


_FwdCtx = namedtuple("_FwdCtx", "cos sin mask pos_ids")


def _make_fwd(args, body, lm_head, cur_x, ctx):
    """Build the fixed-window forward closure. M2.7 layers return a tuple; M3 layers return hidden.

    ``ctx`` is a ``_FwdCtx`` bundling the (cos, sin, mask, pos_ids) capture tensors.
    """
    is_m27 = args.variant == "m27"

    @torch.no_grad()
    def fwd():
        hidden = body.embed_tokens(cur_x)
        for layer in body.layers:
            out = layer(hidden, position_embeddings=(ctx.cos, ctx.sin), attention_mask=ctx.mask,
                        position_ids=ctx.pos_ids, past_key_values=None, use_cache=False, cache_position=None)
            hidden = (out[0] if isinstance(out, tuple) else out) if is_m27 else out
        return lm_head(body.norm(hidden))
    return fwd


def _capture(fwd):
    """Warm fwd on a side stream, then NPUGraph-capture it; return (graph, captured_output)."""
    torch.npu.synchronize()
    side_stream = torch.npu.Stream()
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        for _ in range(3):
            fwd()
    torch.npu.current_stream().wait_stream(side_stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fwd()
    return graph, out


def run_graph(args, dev):
    """NPUGraph-captured E2E arm: a fixed-window forward is captured once and replayed per generated
    token (host-side argmax + window shift between replays). Reports decode-generate tok/s. The static
    MoE route sends every token to the first top_k experts, so only those are dequantized onto the die.
    ``graph`` runs a vectorized FFN, ``pypto+graph`` the fused grouped GEMM.
    """
    spec = _VARIANTS[args.variant]
    use_pypto = args.use_pypto
    mode = "pypto+graph" if use_pypto else "graph"
    logger.info("\n%s\n  %s E2E graph-replay -- %s\n%s",
                "=" * 70, spec["display"], mode.upper(), "=" * 70)

    model, cfg, n_patched = load_variant_model(args, dev, streaming=False)
    body, lm_head = model.model, model.lm_head
    window = args.graph_window
    logger.info("  loaded %d layers (patched %d MoE), window=%d", cfg.num_hidden_layers, n_patched, window)
    if args.variant == "m27":
        _install_static_moe_m27(body, dev, use_pypto)
    else:
        _install_static_moe_m3(body, use_pypto)

    cur_x = torch.randint(0, cfg.vocab_size, (1, window), device=dev)
    pos_ids = torch.arange(window, device=dev).unsqueeze(0)
    neg = torch.finfo(torch.bfloat16).min
    mask = torch.triu(torch.full((window, window), neg, device=dev, dtype=torch.bfloat16), 1)
    mask = mask.view(1, 1, window, window)
    with torch.no_grad():
        cos, sin = body.rotary_emb(torch.zeros(1, window, 1, device=dev, dtype=torch.bfloat16), pos_ids)
    cos, sin = cos.contiguous(), sin.contiguous()

    fwd = _make_fwd(args, body, lm_head, cur_x, _FwdCtx(cos, sin, mask, pos_ids))
    graph, captured_logits = _capture(fwd)
    logger.info("  [graph] capture OK")

    num_new = args.max_new_tokens

    def gen_once():
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(num_new):
            graph.replay()
            next_id = captured_logits[0, -1].argmax()
            cur_x[:, :-1] = cur_x[:, 1:].clone()
            cur_x[0, -1] = next_id
        torch.npu.synchronize()
        return time.perf_counter() - start

    for _ in range(args.warmup):
        gen_once()
    times = [gen_once() for _ in range(args.iters)]
    tps_mean, tps_std = _mean_std([num_new / t for t in times])
    peak_mem = torch.npu.max_memory_allocated() / 1024 ** 2

    result = {
        "model": spec["display"], "mode": mode, "device": dev,
        "layers": cfg.num_hidden_layers, "moe_patched": n_patched,
        "graph_window": window, "max_new_tokens": num_new, "warmup": args.warmup, "iters": args.iters,
        "decode_tps_mean": tps_mean, "decode_tps_std": tps_std,
        "time_s": _mean_std(times), "peak_mem_mb": round(peak_mem, 1),
    }
    logger.info("\n--- %s Summary (avg of %d replays of %d tok) ---", mode.upper(), args.iters, num_new)
    logger.info("  decode-generate: %.1f +/- %.1f tok/s", tps_mean, tps_std)
    logger.info("  peak mem: %.0f MB", peak_mem)
    return result


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit("set MODEL_PATH or pass --model-path")
    os.environ.setdefault("PYPTO_VEC_TILE", _VARIANTS[args.variant]["vec_tile"])
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(args.device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(args.device)
    dev = f"npu:{args.device}"

    result = run_graph(args, dev) if args.graph else run_generate(args, dev)

    if args.report_file:
        with open(args.report_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("\nReport saved: %s", args.report_file)


if __name__ == "__main__":
    main()
