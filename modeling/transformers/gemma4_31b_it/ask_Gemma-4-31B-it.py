#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gemma-4-31B-it inference script with optional PyPTO kernel acceleration."""

import logging
import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationStats:
    input_len: int
    output_tokens: int
    gen_time: float
    tokens_per_sec: float
    tokenizer_load_time: float
    model_load_time: float


def setup_pypto(repo_root, model_path):
    """Install patched modeling file and inject PyPTO kernel switch."""
    sys.path.insert(0, os.path.join(repo_root, "src"))

    overlay_dir = tempfile.mkdtemp(prefix="gemma4_pypto_model_")
    atexit.register(lambda: shutil.rmtree(overlay_dir, ignore_errors=True))

    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        dst = os.path.join(overlay_dir, name)
        if name in {"config.json", "configuration_gemma4.py", "modeling_gemma4.py"}:
            continue
        os.symlink(src, dst)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "configuration_gemma4.Gemma4Config",
        "AutoModelForImageTextToText": "modeling_gemma4.Gemma4ForConditionalGeneration",
    }
    with open(os.path.join(overlay_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Install the modified modeling/configuration files into a private overlay directory.
    patched_src = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                               "gemma4_31b_it", "modeling_gemma4.py")
    patched_cfg = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                               "gemma4_31b_it", "configuration_gemma4.py")
    shutil.copy2(patched_src, os.path.join(overlay_dir, "modeling_gemma4.py"))
    shutil.copy2(patched_cfg, os.path.join(overlay_dir, "configuration_gemma4.py"))

    cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.isdir(cache_root):
        for name in os.listdir(cache_root):
            if "gemma" in name.lower() or "gemma4" in name.lower():
                shutil.rmtree(os.path.join(cache_root, name), ignore_errors=True)
    logging.info(f"[PyPTO] Installed patched modeling overlay at {overlay_dir}")

    # Import real kernel adapter module and register under expected name
    from ops.pypto_tensor import gemma4_31b_it as gemma4_kernels
    gemma4_kernels.USE_PTO_SOFTMAX = True
    gemma4_kernels.USE_PTO_GQA = True
    sys.modules["gemma4_pto_kernels"] = gemma4_kernels
    logging.info("[PyPTO] Attention SoftMax and GQA Decode kernels enabled")
    return overlay_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-4-31B-it inference")
    parser.add_argument("--prompt", type=str, default="Explain machine learning in simple terms.")
    parser.add_argument("--device", type=str, default="0", help="NPU device ID")
    parser.add_argument("--model-path", type=str, required=True, help="Model weight path")
    parser.add_argument("--sentence_file", type=str, default=None, help="Input prompts file")
    parser.add_argument("--output_length", type=int, default=100, help="Max new tokens")
    parser.add_argument("--use_pypto", action="store_true", help="Enable PyPTO fused kernels")
    parser.add_argument("--graph", action="store_true",
                        help="NPUGraph-capture decode arm (captures a decode step, replays output_length times)")
    parser.add_argument("--ctx", type=int, default=256, help="warmed context length for --graph")
    parser.add_argument("--report_file", type=str, default=None, help="JSON report output path")
    parser.add_argument("--show_outputs", action="store_true", help="Print generated text")
    return parser.parse_args()


def resolve_load_path(args):
    if args.use_pypto:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        return setup_pypto(repo_root, args.model_path)
    return args.model_path


def load_tokenizer_and_model(load_path, device_str):
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    tok_time = time.time() - t0

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        device_map=device_str,
        trust_remote_code=True,
    )
    model_time = time.time() - t0
    return tokenizer, model, tok_time, model_time


def read_prompt(args):
    if args.sentence_file and os.path.exists(args.sentence_file):
        with open(args.sentence_file, "r") as f:
            return f.read().strip()
    return args.prompt


def generate_once(args, tokenizer, model, device_str):
    prompt = read_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.output_length)
    gen_time = time.time() - t0

    output_tokens = outputs.shape[1] - input_len
    tps = output_tokens / gen_time if gen_time > 0 else 0
    return inputs, outputs, output_tokens, gen_time, tps


def print_summary(args, stats):
    logging.info(f"\n{'='*60}")
    logging.info(f"Model       : {args.model_path}")
    logging.info(f"PyPTO       : {'ON' if args.use_pypto else 'OFF'}")
    logging.info(f"Input tokens: {stats.input_len}")
    logging.info(f"Output tokens: {stats.output_tokens}")
    logging.info(f"Generate time: {stats.gen_time:.2f}s")
    logging.info(f"Tokens/sec   : {stats.tokens_per_sec:.1f}")
    logging.info(f"Tokenizer load: {stats.tokenizer_load_time:.2f}s")
    logging.info(f"{'='*60}")
    logging.info(f"{'='*60}")


def maybe_write_report(args, stats):
    if args.report_file:
        report = {
            "model": args.model_path,
            "pypto": args.use_pypto,
            "input_tokens": stats.input_len,
            "output_tokens": stats.output_tokens,
            "generate_time_s": stats.gen_time,
            "tokens_per_sec": stats.tokens_per_sec,
            "tokenizer_load_s": stats.tokenizer_load_time,
            "model_load_s": stats.model_load_time,
        }
        with open(args.report_file, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"Report saved to {args.report_file}")


class _CaptureCache:
    """Fixed-slot KV cache (no host ops) so a decode step captures and replays stably."""

    def __init__(self, head_dims, num_kv, maxlen, device, num_layers):
        self.keys = [
            torch.zeros(1, num_kv[i], maxlen, head_dims[i], device=device, dtype=torch.bfloat16)
            for i in range(num_layers)
        ]
        self.values = [
            torch.zeros(1, num_kv[i], maxlen, head_dims[i], device=device, dtype=torch.bfloat16)
            for i in range(num_layers)
        ]
        self.write_pos = None
        self.length = 0

    def update(self, key, value, layer_idx, *args, **kwargs):
        self.keys[layer_idx].index_copy_(2, self.write_pos, key)
        self.values[layer_idx].index_copy_(2, self.write_pos, value)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_mask_sizes(self, q_length, layer_idx=0):
        return self.keys[0].shape[2], 0


def _build_decode_state(body, cfg, ctx_len, gen_n, device):
    """Build the capture-safe KV cache + fixed decode tensors at a warmed context of ctx_len."""
    num_layers = len(body.layers)
    head_dims = [body.layers[i].self_attn.head_dim for i in range(num_layers)]
    num_kv = [body.layers[i].self_attn.k_proj.out_features // head_dims[i] for i in range(num_layers)]
    cache = _CaptureCache(head_dims, num_kv, ctx_len + gen_n + 1, device, num_layers)
    torch.manual_seed(0)
    with torch.no_grad():                     # warm context slots (content irrelevant for perf)
        for i in range(num_layers):
            cache.keys[i].normal_(0, 0.1)
            cache.values[i].normal_(0, 0.1)
    cache.write_pos = torch.tensor([ctx_len], device=device)
    cache.length = ctx_len
    zmask = torch.zeros(1, 1, 1, ctx_len + gen_n + 1, device=device, dtype=torch.bfloat16)
    return types.SimpleNamespace(
        cache=cache,
        dec_id=torch.randint(0, int(cfg.vocab_size), (1, 1), device=device),
        cache_pos=torch.tensor([ctx_len], device=device),
        pos_ids=torch.tensor([[ctx_len]], device=device),
        mask_dict={"full_attention": zmask, "sliding_attention": zmask},
    )


def _report_decode(args, mode, ctx_len, gen_n, best):
    """Print the gemma4 decode result and optionally write the JSON report."""
    tps = round(gen_n / best, 2)
    sep = "=" * 60
    logging.info(f"\n{sep}\nGemma-4-31B-it  {mode}  decode tok/s: {tps}  ({best / gen_n * 1000:.2f} ms/tok)\n{sep}")
    if args.report_file:
        report = {"model": args.model_path, "mode": mode, "ctx": ctx_len, "output_length": gen_n,
                  "decode_tok_s": tps, "ms_per_tok": round(best / gen_n * 1000, 3)}
        with open(args.report_file, "w") as handle:
            json.dump(report, handle, indent=2)
        logging.info(f"Report saved to {args.report_file}")


def run_graph(args):
    """NPUGraph-capture decode arm: capture a steady-state decode step (seq=1) once and replay it
    for output_length tokens (cache write-slot advanced in place). Reports decode tok/s.
    """
    device = f"npu:{args.device}"
    os.environ.setdefault("TILE_FWK_DEVICE_ID", args.device)
    load_path = resolve_load_path(args)            # installs the PyPTO overlay when --use_pypto
    torch.npu.set_device(int(args.device))
    ctx_len, gen_n = args.ctx, args.output_length
    if ctx_len + gen_n >= 1024:
        raise ValueError("keep ctx + output_length < sliding_window for the all-zeros-mask shortcut")

    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        load_path, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="eager").eval()
    cfg = model.config.get_text_config()
    body = next(m for m in model.modules()
                if all(hasattr(m, attr) for attr in ("layers", "rotary_emb", "embed_tokens")))
    lm_head = model.get_output_embeddings()
    st = _build_decode_state(body, cfg, ctx_len, gen_n, device)

    def step():
        with torch.no_grad():
            out = body(input_ids=st.dec_id, past_key_values=st.cache, attention_mask=st.mask_dict,
                       cache_position=st.cache_pos, position_ids=st.pos_ids, use_cache=True)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return lm_head(hidden[:, -1:, :]).argmax(-1)

    def advance(next_id):
        st.dec_id.copy_(next_id)
        st.cache.write_pos.add_(1)
        st.cache_pos.add_(1)
        st.pos_ids.add_(1)

    side_stream = torch.npu.Stream()
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        for _ in range(3):
            step()
    torch.npu.current_stream().wait_stream(side_stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_next = step()
    logging.info("[graph] capture OK")

    def run_generate():
        st.cache.write_pos.fill_(ctx_len)
        st.cache_pos.fill_(ctx_len)
        st.pos_ids.fill_(ctx_len)
        torch.npu.synchronize()
        start = time.time()
        for _ in range(gen_n):
            graph.replay()
            advance(captured_next)
        torch.npu.synchronize()
        return time.time() - start

    for _ in range(2):
        run_generate()
    times = [run_generate() for _ in range(5)]
    best = sum(times) / len(times)            # report the AVERAGE over iters, not best-of
    mode = "pypto+graph" if args.use_pypto else "graph"
    _report_decode(args, mode, ctx_len, gen_n, best)


def main():
    args = parse_args()
    if args.graph:
        run_graph(args)
        return
    device_str = f"npu:{args.device}"
    os.environ.setdefault("TILE_FWK_DEVICE_ID", args.device)
    load_path = resolve_load_path(args)
    tokenizer, model, tok_time, model_time = load_tokenizer_and_model(load_path, device_str)
    inputs, outputs, output_tokens, gen_time, tps = generate_once(args, tokenizer, model, device_str)
    stats = GenerationStats(
        input_len=inputs["input_ids"].shape[1],
        output_tokens=output_tokens,
        gen_time=gen_time,
        tokens_per_sec=tps,
        tokenizer_load_time=tok_time,
        model_load_time=model_time,
    )
    print_summary(args, stats)
    if args.show_outputs:
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"\nGenerated:\n{text}")
    maybe_write_report(args, stats)


if __name__ == "__main__":
    main()
