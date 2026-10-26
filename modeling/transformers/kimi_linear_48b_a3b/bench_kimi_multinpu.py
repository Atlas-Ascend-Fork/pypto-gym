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
"""One-process-per-NPU, pipeline-parallel benchmark for Kimi-Linear-48B-A3B.

Self-contained in this repo (no external/archive references): imports the committed
``pypto_gym.transformers.kimi_linear_48b_a3b`` modeling + the
``kimi_linear_48b_a3b_pto_kernels`` package, and wires PyPTO via the same
``sys.modules`` + ``USE_PTO_KDA`` mechanism as ``ask_Kimi-Linear-48B-A3B.py``.

Supports two modes:
  --mode prefill: Measure prefill performance (prompt processing)
  --mode decode:  Measure decode performance (token generation)

WHY THIS EXISTS: the single-process ``ask_*.py`` (device_map) binds PyPTO's JIT
kernel to ONE NPU per process, so only ~6/20 KDA layers are accelerated. This bench
runs ONE PROCESS PER NPU; each rank holds a contiguous pipeline stage and sets
``TILE_FWK_DEVICE_ID=<local_rank>`` BEFORE importing the kernels, so PyPTO binds
per-NPU and EVERY rank's KDA layers fire -> full (20/20) coverage. Hidden state
streams rank->rank over HCCL. Same pattern as upstream ``minimax_m27`` /
``bench_minimax_m27.py``.

Launch with torchrun (inside the container, Ascend env sourced):

  # SMOKE — random weights (no checkpoint needed), 2 NPUs, 4 layers, seq 80:
  torchrun --nproc_per_node=2 bench_kimi_multinpu.py --mode prefill \
    --random-weights --layers 4 --seq 80 --iters 2 --pypto

  # REAL — load the checkpoint, full model, 4 NPUs, measure prefill:
  MODEL_PATH=/data/models/Kimi-Linear-48B-A3B-Instruct \
    torchrun --nproc_per_node=4 bench_kimi_multinpu.py --mode prefill --pypto --seq 300 --iters 5 --report-file out.json

  # decode mode:
  MODEL_PATH=/data/models/Kimi-Linear-48B-A3B-Instruct \
    torchrun --nproc_per_node=4 bench_kimi_multinpu.py --mode decode --pypto --report-file decode.json

Compare PyPTO vs baseline by running once with --pypto and once without.

NOTE: --random-weights produces meaningless logits (it exists to validate the
harness + measure kernel/pipeline timing without the 96GB checkpoint). Real
perf/quality numbers require OMITTING --random-weights (the checkpoint is then
loaded per-stage from MODEL_PATH) and running the full model.
"""
import argparse
import glob
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(message)s")

_RANK = 0


def log(msg):
    logging.info("[rank %s] %s", _RANK, msg)

# --- bootstrap the in-repo package onto sys.path (walk up to the dir holding src/) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_P = _HERE
while _P != "/" and not os.path.isdir(os.path.join(_P, "src", "pypto_gym")):
    _P = os.path.dirname(_P)
_REPO_ROOT = _P
sys.path.insert(0, _REPO_ROOT)  # top-level ops package (kernels) lives at the repo root
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "pypto_gym", "transformers", "kimi_linear_48b_a3b"))

# committed config.json (used for --random-weights so no weights dir is needed)
_REPO_CONFIG = os.path.join(
    _REPO_ROOT, "src", "pypto_gym", "transformers", "kimi_linear_48b_a3b", "config.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/models/Kimi-Linear-48B-A3B-Instruct")

# per-process KDA coverage counter (incremented by the wrapped kernel wrappers)
_COVER = {"pypto_fused_prefill": 0, "fallback": 0}


@dataclass
class Ctx:
    """Per-rank pipeline context (topology + model), threaded through the stage helpers."""
    cfg: object
    stage: list
    rank: int
    world: int
    dev: str
    model: object = None
    seq: int = 0
    stage_run: object = None  # captured NPU-graph stage closure (--graph), else None
    batch: int = 1
    prompt_len: int = 128
    tokens: int = 64
    warmup: int = 2


@dataclass
class RunModeParams:
    """Parameters for prefill/decode mode execution."""
    args: object
    rank: int
    world: int
    local: int
    dev: str
    cfg: object
    stage: list


@dataclass
class StaticRouteParams:
    """Parameters for static route installation."""
    cfg: object
    model: object
    lo: int
    hi: int
    dev: str
    requested_active: int


def _stage_bounds(num_layers, world):
    """Contiguous [lo, hi) layer range owned by each rank (rank0 also embed, last
    rank also norm + lm_head)."""
    per = (num_layers + world - 1) // world
    return [(r * per, min((r + 1) * per, num_layers)) for r in range(world)]


def _owned_prefixes(stage, rank, world):
    lo, hi = stage[rank]
    pref = [f"model.layers.{i}." for i in range(lo, hi)]
    if rank == 0:
        pref.append("model.embed_tokens.")
    if rank == world - 1:
        pref.append("model.norm.")
        pref.append("lm_head.")
    return pref


def _assign(model, qualified_name, tensor):
    parts = qualified_name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    leaf = parts[-1]
    if leaf in mod._parameters and mod._parameters[leaf] is not None:
        mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in mod._buffers and mod._buffers[leaf] is not None:
        mod._buffers[leaf] = tensor
    else:
        setattr(mod, leaf, torch.nn.Parameter(tensor, requires_grad=False))


def _materialize_owned(model, prefixes, dev, random_weights):
    """Move this rank's owned params/buffers off ``meta`` onto ``dev``.

    random_weights: fill with small normal noise (smoke; no checkpoint).
    else: load the matching tensors from the MODEL_PATH safetensors shards in
    their native checkpoint dtype (linear=bf16, KDA gate A_log/dt_bias=fp32).
    """
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    owned = [n for n in named if any(n.startswith(p) for p in prefixes)]
    if random_weights:
        torch.manual_seed(0)
        for n in owned:
            t = named[n]
            new = torch.empty(t.shape, dtype=(torch.float32 if t.dtype == torch.float32 else torch.bfloat16),
                              device=dev)
            new.normal_(0.0, 0.02)
            _assign(model, n, new)
        return len(owned)
    from safetensors import safe_open
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for n in owned:
        if n in weight_map:
            by_shard.setdefault(weight_map[n], []).append(n)
    loaded = 0
    for shard, names in by_shard.items():
        with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
            for n in names:
                _assign(model, n, f.get_tensor(n).to(dev))
                loaded += 1
    return loaded


def _truncate_layers(cfg, n):
    """Shrink the config to n decoder layers (smoke), keeping only the 1-based
    kda/full-attn layer indices that survive the truncation."""
    cfg.num_hidden_layers = n
    la = cfg.linear_attn_config
    if not isinstance(la, dict):
        return
    for key in ("kda_layers", "full_attn_layers"):
        if la.get(key):
            la[key] = [i for i in la[key] if i <= n]


def _build_config(args):
    from pypto_gym.transformers.kimi_linear_48b_a3b.configuration_kimi import KimiLinearConfig
    src = _REPO_CONFIG if args.random_weights else os.path.join(MODEL_PATH, "config.json")
    with open(src) as f:
        raw = json.load(f)
    cfg = KimiLinearConfig(**raw)
    if args.layers is not None and args.layers < cfg.num_hidden_layers:
        _truncate_layers(cfg, args.layers)
    cfg._attn_implementation = "eager"  # manual additive MLA mask below
    return cfg


def _build_model(ctx, args):
    from pypto_gym.transformers.kimi_linear_48b_a3b.modeling_kimi import KimiLinearForCausalLM
    with torch.device("meta"):
        model = KimiLinearForCausalLM(ctx.cfg)
    model.eval()
    model.config._attn_implementation = "eager"
    n = _materialize_owned(model, _owned_prefixes(ctx.stage, ctx.rank, ctx.world), ctx.dev,
                           args.random_weights)
    log(f"materialized {n} owned tensors ({'random' if args.random_weights else 'checkpoint'})")
    return model


def _wire_pypto(local_rank):
    """Register the committed kernels package under the name modeling_kimi expects, enable
    USE_PTO_KDA, and wrap the fused prefill operator to count pypto-vs-fallback. Binds
    PyPTO to THIS rank's NPU (TILE_FWK_DEVICE_ID was set to local_rank before import)."""
    import ops.pypto_tensor.kimi_linear_48b_a3b as pto
    sys.modules["kimi_linear_48b_a3b_pto_kernels"] = pto
    _orig = pto.kda_fused_prefill

    def counting_fused(*a, **kw):
        try:
            r = _orig(*a, **kw)
            _COVER["pypto_fused_prefill"] += 1
            return r
        except NotImplementedError:
            _COVER["fallback"] += 1
            raise

    pto.kda_fused_prefill = counting_fused
    pto.USE_PTO_KDA = True
    log("PyPTO wired: USE_PTO_KDA=True (fused prefill + fused decode), "
        "kernel bound to this rank's NPU")


def _mla_mask(t_step, kv_len, dev):
    """[1,1,T,kv] additive causal mask for eager MLA (prefill, abs_pos=0)."""
    neg = torch.finfo(torch.bfloat16).min
    q_pos = torch.arange(t_step, device=dev).view(t_step, 1)
    k_pos = torch.arange(kv_len, device=dev).view(1, kv_len)
    mask = torch.zeros(1, 1, t_step, kv_len, dtype=torch.bfloat16, device=dev)
    return mask.masked_fill((k_pos > q_pos).view(1, 1, t_step, kv_len), neg)


def precompute_static_routing(width, top_k, num_experts, num_tokens, dev):
    """Host-side perm/inv/offsets for a deterministic controlled route (port of
    minimax precompute_static_routing). `width` = number of distinct experts the
    tokens are spread over. ALL routing (the argsort that runs on AICPU and
    synchronizes, plus bincount) is done HERE on the host, OUTSIDE any captured
    region, leaving only device gather/matmul/scatter inside the graph.

    Returns (tok_src, inv, offs):
      tok_src : [N*K] device int — token index feeding each sorted slot
      inv     : [N*K] device int — inverse permutation (sorted -> route order)
      offs    : host list of (expert_id, start_offset, count) over fixed slices
    """
    flat = (torch.arange(num_tokens * top_k) % width)        # route slot -> expert
    perm = flat.argsort()                                     # host argsort (no graph)
    tok_src = (perm // top_k).to(dev)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(num_tokens * top_k)
    counts = torch.bincount(flat[perm], minlength=num_experts)
    offs, off = [], 0
    for e in range(num_experts):
        c = int(counts[e])
        if c:
            offs.append((e, off, c))
            off += c
    return tok_src, inv.to(dev), offs


def make_controlled_gate(top_k, width):
    """Drop-in replacement for KimiMoEGate.forward returning a deterministic route.

    Returns (topk_idx, topk_weight) for the static route (slot j of token i -> expert
    (i*top_k+j) % width). Bypasses the sigmoid/group/e_score_bias/scaling — fine for a
    throughput proxy (logits intentionally won't match). topk_weight = 1/top_k.
    """
    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        n = bsz * seq_len
        dev = hidden_states.device
        ar = torch.arange(n * top_k, device=dev).view(n, top_k)
        idx = (ar % width).to(torch.int64)
        w = torch.full((n, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)
        return idx, w
    return forward


def make_static_moe_infer(tok_src, inv, offs, top_k):
    """Drop-in replacement for KimiSparseMoeBlock.moe_infer for a fixed route.

    The stock moe_infer host-syncs three ways (argsort :834, .cpu().numpy() :837,
    data-dependent per-expert loop :841). Here all of that is precomputed on the
    host (tok_src/inv/offs); inside, only device ops remain: index_select gather ->
    per-expert FFN over HOST-FIXED contiguous slices -> index_select scatter ->
    weighted sum. No host sync, no AICPU op -> NPU-graph capturable.
    """
    def moe_infer(self, x, topk_ids, topk_weight):
        hidden_size = x.shape[-1]
        sorted_x = x.index_select(0, tok_src)            # [N*K, H]
        out = torch.empty_like(sorted_x)
        for e, off, c in offs:
            expert = self.experts[e + self.ep_rank * self.experts_per_rank]
            out[off:off + c] = expert(sorted_x[off:off + c])
        combined = out.index_select(0, inv).view(-1, top_k, hidden_size)
        return (combined.type(topk_weight.dtype)
                * topk_weight.reshape(-1, top_k).unsqueeze(-1)).sum(1).type(x.dtype)
    return moe_infer


def _install_static_moe(ctx, width):
    """Patch every owned MoE block in this rank's stage with the controlled gate +
    static moe_infer (skips dense layers, e.g. layer 0 < first_k_dense_replace)."""
    top_k = ctx.cfg.num_experts_per_token
    num_experts = ctx.cfg.num_experts
    tok_src, inv, offs = precompute_static_routing(width, top_k, num_experts, ctx.seq, ctx.dev)
    lo, hi = ctx.stage[ctx.rank]
    patched = 0
    gate_fwd = make_controlled_gate(top_k, width)
    infer_fwd = make_static_moe_infer(tok_src, inv, offs, top_k)
    for i in range(lo, hi):
        layer = ctx.model.model.layers[i]
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        moe.gate.forward = gate_fwd.__get__(moe.gate)
        moe.moe_infer = infer_fwd.__get__(moe)
        patched += 1
    return patched, len(offs)


def _install_expert_counter(ctx):
    """Wrap each owned MoE block's gate to record the number of DISTINCT experts that
    real routing lights up per layer (port of minimax _run_count_experts hook). Returns
    a list that fills with (layer_idx, n_distinct) after a forward pass."""
    active_log = []
    lo, hi = ctx.stage[ctx.rank]
    for i in range(lo, hi):
        layer = ctx.model.model.layers[i]
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        orig = moe.gate.forward

        def counting(self, hidden_states, _orig=orig, _li=i):
            idx, w = _orig(hidden_states)
            active_log.append((_li, int(idx.reshape(-1).unique().numel())))
            return idx, w
        moe.gate.forward = counting.__get__(moe.gate)
    return active_log


def _make_stage_forward(ctx, t):
    """Comm-free local-compute closure with constants (mask/position_ids/cache_position)
    hoisted out and use_cache=False (cache_params=None -> KDA functional, no cache
    writes/sync). Captures cleanly once the static MoE route is installed."""
    cache_position = torch.arange(t, device=ctx.dev)
    position_ids = cache_position.unsqueeze(0)
    causal = _mla_mask(t, t, ctx.dev)
    lo, hi = ctx.stage[ctx.rank]

    def stage_forward(hidden):
        for i in range(lo, hi):
            layer = ctx.model.model.layers[i]
            mask = None if getattr(layer, "is_linear_attn", False) else causal
            hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                           past_key_values=None, use_cache=False, cache_position=cache_position)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        if ctx.rank == ctx.world - 1:
            hidden = ctx.model.model.norm(hidden)
        return hidden

    return stage_forward


def _install_static_route(ctx, args, tag="static"):
    """Install the controlled gate + static moe_infer for the requested route on every
    owned MoE block. Shared by the graph path (capture) and the NO-graph path (eager,
    isolates host-sync removal from graph capture). Returns (width, patched, n_active)."""
    width = ctx.cfg.num_experts if args.route == "uniform" else max(
        ctx.cfg.num_experts_per_token, min(args.active, ctx.cfg.num_experts))
    patched, n_active = _install_static_moe(ctx, width)
    log(f"[{tag}] static-route MoE installed on {patched} MoE block(s), "
        f"route={args.route} width={width} active_experts/layer={n_active}")
    return width, patched, n_active


def _install_static_route_decode(params):
    """Fixed host-precomputed MoE route on this rank's MoE blocks (decode mode)."""
    cfg, model, lo, hi, dev, requested_active = (
        params.cfg, params.model, params.lo, params.hi, params.dev, params.requested_active)
    batch = 1
    top_k = cfg.num_experts_per_token
    reachable = batch * top_k
    width = min(requested_active or reachable, reachable, cfg.num_experts)
    if requested_active and requested_active > reachable:
        log(f"[route] --active {requested_active} clamped to {width}: "
            f"decode step with batch={batch}, top_k={top_k} has only {reachable} slots")

    tok_src, inv, offs = precompute_static_routing(width, top_k, cfg.num_experts,
                                                    batch, dev)
    gate_fwd = make_controlled_gate(top_k, width)
    infer_fwd = make_static_moe_infer(tok_src, inv, offs, top_k)
    n = 0
    for i in range(lo, hi):
        moe = getattr(model.model.layers[i], "block_sparse_moe", None)
        if moe is None:
            continue
        moe.gate.forward = gate_fwd.__get__(moe.gate)
        moe.moe_infer = infer_fwd.__get__(moe)
        n += 1
    return width


def _make_graph_stage_run(ctx, args):
    """Install static routing, build the comm-free stage closure, warm it up (3x),
    then NPU-graph-capture it. Returns stage_run(hidden) -> hidden replaying the
    captured graph. HCCL recv/send stay OUTSIDE this (in _prefill_once)."""
    _install_static_route(ctx, args, tag="graph")

    # PyPTO + graph: the fused prefill path is capture-clean (stream-ordered,
    # no separate graph entrypoint), so nothing to flip here — the counting
    # wrapper from _wire_pypto also covers the warmup/capture invocations.

    stage_forward = _make_stage_forward(ctx, args.seq)
    hin = torch.zeros(1, args.seq, ctx.cfg.hidden_size, dtype=torch.bfloat16, device=ctx.dev)
    with torch.no_grad():
        for _ in range(3):
            stage_forward(hin)
    torch.npu.synchronize()
    log("[graph] warmup (3x) OK, capturing ...")
    gobj = torch.npu.NPUGraph()
    with torch.no_grad(), torch.npu.graph(gobj):
        hout = stage_forward(hin)
    torch.npu.synchronize()
    log("[graph] capture OK")

    def stage_run(hidden):
        hin.copy_(hidden)
        gobj.replay()
        return hout

    return stage_run


def _run_stage(ctx, hidden, cache, t):
    cache_position = torch.arange(t, device=ctx.dev)
    position_ids = cache_position.unsqueeze(0)
    causal = _mla_mask(t, t, ctx.dev)
    lo, hi = ctx.stage[ctx.rank]
    for i in range(lo, hi):
        layer = ctx.model.model.layers[i]
        mask = None if getattr(layer, "is_linear_attn", False) else causal
        hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                       past_key_values=cache, use_cache=True, cache_position=cache_position)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
    if ctx.rank == ctx.world - 1:
        hidden = ctx.model.model.norm(hidden)
        logits = ctx.model.lm_head(hidden[:, -1:])
        return hidden, logits
    return hidden, None


def _new_cache(model, cfg):
    mod = sys.modules[type(model).__module__]
    return mod.KimiDynamicCache(config=cfg)


def _prefill_once(ctx, input_ids, t):
    if ctx.rank == 0:
        hidden = ctx.model.model.embed_tokens(input_ids).contiguous()
    else:
        hidden = torch.empty(1, t, ctx.cfg.hidden_size, dtype=torch.bfloat16, device=ctx.dev)
        dist.recv(hidden, src=ctx.rank - 1)
    # graph path: replay the captured comm-free stage (HCCL stays outside, here).
    # eager path: run the stage normally (with a fresh dynamic cache).
    if ctx.stage_run is not None:
        hidden = ctx.stage_run(hidden)
    else:
        cache = _new_cache(ctx.model, ctx.cfg)
        hidden, _ = _run_stage(ctx, hidden, cache, t)
    if ctx.rank < ctx.world - 1:
        # stream the stage output in a fixed dtype so the send/recv dtypes match
        # across ranks (the stage may emit fp32; the recv buffer is bf16).
        dist.send(hidden.to(torch.bfloat16).contiguous(), dst=ctx.rank + 1)


def _parse_computing_us(prof_dir):
    """Return Computing(us) from step_trace_time.csv under prof_dir — the same
    torch_npu.profiler NPU-compute metric the .archive runs used (skill method 4)."""
    import csv
    hits = glob.glob(os.path.join(prof_dir, "**", "step_trace_time.csv"), recursive=True)
    if not hits:
        return None
    with open(sorted(hits)[0]) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    for key in rows[-1]:
        if key.strip().lower().startswith("computing"):
            try:
                return float(rows[-1][key])
            except (ValueError, TypeError):
                return None
    return None


def _profile_compute_us(ctx, input_ids, t, prof_dir):
    """Profile two prefill passes and return THIS rank's NPU Computing(us) via
    torch_npu.profiler (step_trace, ProfilerLevel.Level0). Separate from the wall
    loop, so the reported wall stays un-profiled (profiling inflates wall)."""
    import torch_npu
    os.makedirs(prof_dir, exist_ok=True)
    sched = torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU],
        schedule=sched,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
        experimental_config=exp,
    ) as prof:
        for _ in range(2):
            dist.barrier()
            _prefill_once(ctx, input_ids, t)
            torch.npu.synchronize()
            prof.step()
    dist.barrier()
    return _parse_computing_us(prof_dir)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode"], default="prefill",
                    help="benchmark mode: prefill (prompt processing) or decode (token generation)")
    ap.add_argument("--seq", type=int, default=128, help="prefill sequence length (use >64 to hit the chunk kernel)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--layers", type=int, default=None, help="limit total decoder layers (smoke)")
    ap.add_argument("--pypto", action="store_true",
                    help="enable PyPTO fused KDA kernels (prefill chunk + decode fused)")
    ap.add_argument("--graph", action="store_true",
                    help="NPU-graph-capture the per-die comm-free compute stage (static-route "
                         "MoE + use_cache=False) and replay it (eager KDA only; requires --route, --seq>64)")
    ap.add_argument("--route", choices=["uniform", "fixed"], default=None,
                    help="static (graph-capturable) MoE route: uniform=spread over all experts, "
                         "fixed=exactly --active experts")
    ap.add_argument("--active", type=int, default=178,
                    help="for --route fixed: number of experts to activate (178 = seq=300 measured "
                         "natural distinct-active-experts/layer; see --count-experts)")
    ap.add_argument("--count-experts", action="store_true",
                    help="run ONE real-routing prefill and report distinct active experts/layer "
                         "(the natural load — use the mean as --active). Skips the timing loop.")
    ap.add_argument("--random-weights", action="store_true",
                    help="build from config with random weights (no checkpoint) — harness smoke only")
    ap.add_argument("--report-file", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="also measure NPU Computing(us) via torch_npu.profiler (in a separate "
                         "loop; the reported wall stays un-profiled). Matches the .archive metric.")
    ap.add_argument("--prof-dir", default="/tmp/kimi_mnpu_prof", help="torch_npu.profiler trace dir")
    args = ap.parse_args()
    if args.graph:
        if args.route is None:
            ap.error("--graph requires --route {uniform,fixed} (static route removes the MoE host sync)")
        if args.seq <= 64:
            ap.error("--graph requires --seq > 64 (so KDA takes the capture-clean chunk branch)")
    return args


def _timed_prefill(ctx, args, input_ids):
    """Warmup + timed (un-profiled) prefill loop, plus an optional separate
    profiled loop for NPU Computing(us). Returns (wall_ms, wall_best, coverage, compute_us)."""
    # warmup (first iter also triggers the JIT kernel compile)
    for _ in range(args.warmup):
        _prefill_once(ctx, input_ids, args.seq)
    torch.npu.synchronize()
    dist.barrier()

    times = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        _prefill_once(ctx, input_ids, args.seq)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    wall_ms = sum(times) / len(times)
    wall_best = min(times)
    cov = dict(_COVER)
    log(f"prefill wall mean={wall_ms:.1f}ms best-of-{len(times)}={wall_best:.1f}ms  "
        f"KDA coverage(this rank)={cov}")

    # optional NPU-compute (separate profiled loop; does not affect wall_ms above)
    compute_us = None
    if args.profile:
        mode = "pypto" if args.pypto else "baseline"
        compute_us = _profile_compute_us(ctx, input_ids, args.seq,
                                         os.path.join(args.prof_dir, f"{mode}_r{ctx.rank}"))
        log(f"NPU Computing(us) this rank = {compute_us}")
    return wall_ms, wall_best, cov, compute_us


_PROXY_BANNER = (
    "STATIC-ROUTE THROUGHPUT PROXY — value-infidel, not output-correct, not deployable. "
    "When --route is set the MoE route is FIXED on the host (controlled gate bypasses "
    "the real sigmoid/group/bias router); with --graph the device region is then "
    "NPU-graph-capturable. The logits/outputs intentionally do NOT match the real model; "
    "this measures the throughput of a representative compute shape ONLY.")


def _emit_report(ctx, args, result):
    """All-gather per-rank results to rank 0, which logs + optionally writes the report.

    ``result`` = (wall_ms, wall_best, cov, compute_us) from ``_timed_prefill``.
    """
    wall_ms, wall_best, cov, compute_us = result
    payload = json.dumps({"rank": ctx.rank, "wall_ms": wall_ms, "wall_best_ms": wall_best,
                          "coverage": cov, "compute_us": compute_us})
    gathered = [None] * ctx.world
    dist.all_gather_object(gathered, payload)
    if ctx.rank != 0:
        return
    rows = [json.loads(g) for g in gathered]
    total_fused = sum(r["coverage"].get("pypto_fused_prefill", 0) for r in rows)
    total_fb = sum(r["coverage"]["fallback"] for r in rows)
    slowest = max(r["wall_ms"] for r in rows)
    # best-of-N pipeline wall = slowest stage's best iteration (pipeline gated by the
    # slowest stage); report the max over ranks of each rank's best-of-N.
    slowest_best = max(r["wall_best_ms"] for r in rows)
    comp = [r["compute_us"] for r in rows if r.get("compute_us") is not None]
    total_compute_us = sum(comp) if comp else None
    static = args.route is not None  # static route installed (graph OR no-graph eager)
    out = {
        "mode": "pypto" if args.pypto else "baseline",
        "graph": bool(args.graph),
        "route": args.route if static else None,
        "active_experts": (args.active if (static and args.route == "fixed") else
                           (ctx.cfg.num_experts if (static and args.route == "uniform") else None)),
        "world": ctx.world, "layers": ctx.cfg.num_hidden_layers, "seq": args.seq,
        "iters": args.iters, "weights": "random" if args.random_weights else MODEL_PATH,
        "profiled": bool(args.profile),
        "prefill_wall_ms_pipeline": slowest,
        "prefill_wall_ms_pipeline_best": slowest_best,
        "npu_compute_us_total": total_compute_us,
        "kda_pypto_fused_prefill_total": total_fused, "kda_fallback_total": total_fb,
        "per_rank": rows,
    }
    if static:  # any fixed/uniform fake route (graph OR eager-static) is value-infidel
        out["proxy_banner"] = _PROXY_BANNER
        log("=" * 100)
        log("!! " + _PROXY_BANNER)
        log("=" * 100)
    log(f"RESULT best-of-{args.iters} pipeline wall = {slowest_best:.1f}ms "
        f"(mean {slowest:.1f}ms)")
    log("RESULT " + json.dumps(out))
    if args.report_file:
        with open(args.report_file, "w") as f:
            json.dump(out, f, indent=2)
        log(f"report -> {args.report_file}")


def _emit_decode_report(ctx, args, results, cells):
    payload = json.dumps({"rank": ctx.rank, "results": results})
    gathered = [None] * ctx.world
    dist.all_gather_object(gathered, payload)

    if ctx.rank == 0:
        rows = [json.loads(g) for g in gathered]

        def pipeline_ms(cell):
            vals = [r["results"][cell].get("median_ms") for r in rows
                    if "median_ms" in r["results"].get(cell, {})]
            return max(vals) if len(vals) == ctx.world else None

        agg = {c: pipeline_ms(c) for c, _, _ in cells}

        summary = {
            "mode": "decode",
            "config": {"world": ctx.world, "layers": ctx.cfg.num_hidden_layers,
                       "route": args.route,
                       "weights": "random" if args.random_weights else MODEL_PATH},
            "pipeline_ms_per_token": agg,
            "per_rank": rows,
        }
        if args.report_file:
            with open(args.report_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            log(f"report -> {args.report_file}")


def _run_count_experts(ctx, args, rank, world, dev):
    """Measure distinct active experts/layer via one real-routing forward; log on rank 0."""
    cfg = ctx.cfg
    active_log = _install_expert_counter(ctx)
    torch.manual_seed(1234 + rank)
    input_ids = (torch.randint(0, cfg.vocab_size, (1, args.seq), device=dev)
                 if rank == 0 else None)
    with torch.no_grad():
        _prefill_once(ctx, input_ids, args.seq)
    torch.npu.synchronize()
    cnts = [c for _, c in active_log]
    gathered = [None] * world
    dist.all_gather_object(gathered, json.dumps({"rank": rank, "counts": cnts}))
    if rank != 0:
        return
    allc = [c for g in gathered for c in json.loads(g)["counts"]]
    n_tok_topk = args.seq * cfg.num_experts_per_token
    log(f"[count-experts] MoE layers measured={len(allc)} seq={args.seq} "
        f"tokens*topk={n_tok_topk} num_experts={cfg.num_experts}")
    if allc:
        mean = sum(allc) / len(allc)
        log(f"[count-experts] distinct active experts/layer: "
            f"min={min(allc)} max={max(allc)} mean={mean:.1f} -> use --active {round(mean)}")


# ---------------------------------------------------------------------------
# DECODE MODE FUNCTIONS
# ---------------------------------------------------------------------------

@torch.no_grad()
def _prefill_stage_decode(ctx, input_ids):
    """Run this rank's prefill and return (last_hidden, dyn_cache) - decode mode."""
    from pypto_gym.transformers.kimi_linear_48b_a3b.modeling_kimi import KimiDynamicCache
    cfg, model = ctx.cfg, ctx.model
    lo, hi = ctx.stage[ctx.rank]
    t = ctx.prompt_len
    dev = ctx.dev

    if ctx.rank == 0:
        hidden = model.model.embed_tokens(input_ids).contiguous()
    else:
        hidden = torch.empty(ctx.batch, t, cfg.hidden_size,
                             dtype=torch.bfloat16, device=dev)
        dist.recv(hidden, src=ctx.rank - 1)

    cache = KimiDynamicCache(config=cfg)
    cache_position = torch.arange(t, device=dev)
    neg = torch.finfo(torch.bfloat16).min
    q_pos = cache_position.view(t, 1)
    k_pos = cache_position.view(1, t)
    causal = torch.zeros(1, 1, t, t, dtype=torch.bfloat16, device=dev)
    causal = causal.masked_fill((k_pos > q_pos).view(1, 1, t, t), neg)

    for i in range(lo, hi):
        layer = model.model.layers[i]
        mask = None if getattr(layer, "is_linear_attn", False) else causal
        hidden = layer(hidden, attention_mask=mask,
                       position_ids=cache_position.unsqueeze(0),
                       past_key_values=cache, use_cache=True,
                       cache_position=cache_position)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

    if ctx.rank < ctx.world - 1:
        dist.send(hidden.to(torch.bfloat16).contiguous(), dst=ctx.rank + 1)
    return hidden[:, -1:].contiguous(), cache


def _build_decode_stage(ctx, dyn_cache, use_fused, capture):
    """Seed the static cache from prefill, build the stage closure, capture."""
    import kda_decode_graph as dg
    from ops.pypto_tensor.kimi_linear_48b_a3b.kda import (
        kda_fused_decode_impl as kda_impl)

    cfg, model = ctx.cfg, ctx.model
    lo, hi = ctx.stage[ctx.rank]
    dev, batch = ctx.dev, ctx.batch
    layer_ids = list(range(lo, hi))
    layers = {i: model.model.layers[i] for i in layer_ids}
    max_len = ctx.prompt_len + ctx.tokens + ctx.warmup + 8

    if capture:
        _install_static_route_decode(StaticRouteParams(
            cfg, model, lo, hi, dev, requested_active=None))

    cache = dg.StaticDecodeCache(cfg, layers, batch, max_len, torch.bfloat16,
                                 dev, kda_impl)
    dg.seed_from_prefill(cache, dyn_cache, kda_impl.seed_fused_buffers)

    lb = float(getattr(cfg, "gate_lower_bound", -5.0))
    prepared = (dg.prepare_decode_weights(model, layer_ids, kda_impl, gate_lower_bound=lb)
                if use_fused else {})
    stage = dg.make_decode_stage(dg.DecodeStageParams(
        model, layer_ids, cache, prepared, kda_impl, torch.bfloat16, use_fused))

    is_last = ctx.rank == ctx.world - 1

    def step(hidden):
        h = stage(hidden)
        return model.model.norm(h) if is_last else h

    if not capture:
        return step, cache
    run = dg.make_decode_stage_run(dg.DecodeStageRunParams(
        step, cache, batch, cfg.hidden_size, dev, torch.bfloat16, ctx.warmup))
    return run, cache


@torch.no_grad()
def _decode_once(ctx, step_fn, hidden):
    """One pipelined decode step. HCCL stays outside the captured region."""
    cfg = ctx.cfg
    if ctx.rank > 0:
        buf = torch.empty(ctx.batch, 1, cfg.hidden_size,
                          dtype=torch.bfloat16, device=ctx.dev)
        dist.recv(buf, src=ctx.rank - 1)
        hidden = buf
    out = step_fn(hidden)
    if ctx.rank < ctx.world - 1:
        dist.send(out.to(torch.bfloat16).contiguous(), dst=ctx.rank + 1)
    return out


@torch.no_grad()
def _time_decode(ctx, step_fn, hidden):
    for _ in range(ctx.warmup):
        _decode_once(ctx, step_fn, hidden)
    torch.npu.synchronize()
    dist.barrier()

    times = []
    for _ in range(ctx.tokens):
        torch.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        _decode_once(ctx, step_fn, hidden)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(times),
            "mean_ms": statistics.mean(times),
            "stdev_ms": statistics.pstdev(times),
            "best_ms": min(times)}


def _run_prefill_mode(params):
    """Run prefill benchmark (original bench_kimi_multinpu.py behavior)."""
    args, rank, world, local, dev, cfg, stage = (
        params.args, params.rank, params.world, params.local, params.dev, params.cfg, params.stage)
    ctx = Ctx(cfg=cfg, stage=stage, rank=rank, world=world, dev=dev, seq=args.seq)
    if rank == 0:
        mode = "pypto" if args.pypto else "baseline"
        wts = "random" if args.random_weights else MODEL_PATH
        log(f"layers={cfg.num_hidden_layers} world={world} stages={stage} "
            f"seq={args.seq} mode={mode} weights={wts}")
    ctx.model = _build_model(ctx, args)

    if args.count_experts:
        _run_count_experts(ctx, args, rank, world, dev)
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.graph:
        ctx.stage_run = _make_graph_stage_run(ctx, args)
        # determinism check: two replays of the same input must be bit-identical
        probe = torch.randn(1, args.seq, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        o1 = ctx.stage_run(probe).clone()
        o2 = ctx.stage_run(probe).clone()
        if not torch.equal(o1, o2):
            raise RuntimeError("[graph] replay non-deterministic — captured graph differs")
        log(f"[graph] determinism check OK (out shape={tuple(o1.shape)})")
    elif args.route is not None:
        # NO-graph static route: install the controlled gate + host-precomputed static
        # moe_infer (removes the moe_infer host-sync loop) but run EAGER (no capture).
        # Isolates host-sync removal from graph-capture launch-overhead removal.
        _install_static_route(ctx, args, tag="static")

    torch.manual_seed(1234 + rank)
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq), device=dev) if rank == 0 else None

    _emit_report(ctx, args, _timed_prefill(ctx, args, input_ids))

    dist.barrier()
    dist.destroy_process_group()


def _run_decode_mode(params):
    """Run decode benchmark (fused kernel testing)."""
    args, rank, world, local, dev, cfg, stage = (
        params.args, params.rank, params.world, params.local, params.dev, params.cfg, params.stage)
    if args.pypto:
        import ops.pypto_tensor.kimi_linear_48b_a3b as pto
        sys.modules["kimi_linear_48b_a3b_pto_kernels"] = pto
        pto.USE_PTO_KDA = True

    lo, hi = stage[rank]

    ctx = Ctx(cfg=cfg, stage=stage, rank=rank, world=world, dev=dev, seq=128)
    ctx.model = _build_model(ctx, args)

    if rank == 0:
        n_kda = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i))
        mode = "pypto" if args.pypto else "baseline"
        log(f"MODE=DECODE layers={cfg.num_hidden_layers} (KDA {n_kda}) world={world} "
            f"stages={stage} mode={mode}")
    ctx.warmup = args.warmup

    torch.manual_seed(1234)
    input_ids = (torch.randint(0, cfg.vocab_size, (1, 128),
                               device=dev) if rank == 0 else None)

    if args.graph:
        if args.pypto:
            cells = [("graph_fused", True, True)]
        else:
            cells = [("graph_baseline", False, True)]
    else:
        if args.pypto:
            cells = [("eager_fused", True, False)]
        else:
            cells = [("eager_baseline", False, False)]

    results = {}
    for name, use_fused, capture in cells:
        dist.barrier()
        hidden, dyn = _prefill_stage_decode(ctx, input_ids)
        try:
            step, _ = _build_decode_stage(ctx, dyn, use_fused, capture)
            results[name] = _time_decode(ctx, step, hidden)
            log(f"{name:16s} median {results[name]['median_ms']:.3f} ms/token")
        except Exception as e:
            log(f"{name:16s} FAILED: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        del dyn

    _emit_decode_report(ctx, args, results, cells)

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = _parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    global _RANK
    _RANK = rank
    # bind PyPTO to THIS rank's NPU — must precede the kernels import in _wire_pypto
    os.environ["TILE_FWK_DEVICE_ID"] = str(local)

    import torch_npu  # noqa: F401  registers the NPU backend
    torch.npu.set_device(local)
    dev = f"npu:{local}"
    dist.init_process_group("hccl", rank=rank, world_size=world)

    if args.pypto and args.mode == "prefill":
        _wire_pypto(local)

    cfg = _build_config(args)
    stage = _stage_bounds(cfg.num_hidden_layers, world)

    if args.mode == "prefill":
        _run_prefill_mode(RunModeParams(args, rank, world, local, dev, cfg, stage))
    elif args.mode == "decode":
        _run_decode_mode(RunModeParams(args, rank, world, local, dev, cfg, stage))


if __name__ == "__main__":
    main()
