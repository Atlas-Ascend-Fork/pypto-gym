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

"""openPangu-Embedded-7B single-prompt inference on Ascend NPU.

This is a thin wrapper over the cann-recipes-infer ``PanguEmbeddedRunner`` flow
(``models/pangu-7b/infer.py``) that swaps in the PyPTO-integrated modeling from
``pypto_gym.transformers.openpangu_v5_7b`` and gates the fused decode kernel on
``--use-pto``. The Pangu model is driven by the cann-recipes runner framework
(``executor`` / ``module`` + a YAML), so a ``cann-recipes-infer`` checkout must be
on ``PYTHONPATH`` (``--recipes-path`` or ``$CANN_RECIPES_PATH``).

Prerequisites — run ``source env_setup.sh`` first to set up the environment:
    * CANN toolkit + torch_npu
    * PTO_TILE_LIB_CODE_PATH (pto-isa headers for PyPTO JIT)
    * TILE_FWK_DEVICE_ID, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

Usage:
    source env_setup.sh

    # baseline (PyTorch decoder layers)
    python3 ask_openpangu_v5_7b.py \\
        --recipes-path /path/to/cann-recipes-infer \\
        --model-path  /path/to/openPangu-Embedded-7B \\
        --prompt "你好"

    # PyPTO fused decode kernel
    python3 ask_openpangu_v5_7b.py \\
        --recipes-path /path/to/cann-recipes-infer \\
        --model-path  /path/to/openPangu-Embedded-7B \\
        --prompt "你好" --use-pto
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("openpangu.ask")


def _find_pypto_gym_src():
    """Walk up from this script to locate the pypto-gym ``src`` directory."""
    p = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cand = os.path.join(p, "src")
        if os.path.isdir(cand):
            return cand
        p = os.path.dirname(p)
    raise RuntimeError("Could not locate pypto-gym 'src' directory above " + __file__)


def parse_args():
    parser = argparse.ArgumentParser(description="openPangu-Embedded-7B single-prompt inference")
    parser.add_argument("--model-path", required=True, help="openPangu-Embedded-7B weights directory")
    parser.add_argument("--recipes-path", default=os.environ.get("CANN_RECIPES_PATH"),
                        help="cann-recipes-infer checkout root (provides executor/module/models)")
    parser.add_argument("--yaml", default=None, help="runner YAML (defaults to the reference openpangu_v5_7b.yaml)")
    parser.add_argument("--prompt", default="An attention function can be described as mapping a query and a set "
                                            "of key-value pairs to an output, where the query, keys, values, and "
                                            "output are all vectors. The output is")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override YAML data_config.max_new_tokens")
    parser.add_argument("--use-pto", action="store_true", help="enable the PyPTO fused decode kernel")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warmup run")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.recipes_path or not os.path.isdir(args.recipes_path):
        raise SystemExit(
            "A cann-recipes-infer checkout is required: pass --recipes-path or set CANN_RECIPES_PATH."
        )

    # ---- sys.path: pypto-gym src + cann-recipes root + cann-recipes models/pangu-7b ----
    sys.path.insert(0, _find_pypto_gym_src())
    recipes = os.path.realpath(args.recipes_path)
    sys.path.insert(0, recipes)                                  # executor, module
    sys.path.insert(0, os.path.join(recipes, "models", "pangu-7b"))  # runner_openpangu_dense, models.*

    import torch  # noqa: E402
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise ImportError("torch_npu required (Ascend NPU); source env_setup.sh first.") from exc
    torch.npu.set_device(args.device)

    # ---- enable the PyPTO fused kernel BEFORE the modeling is imported/constructed ----
    if args.use_pto:
        import ops.pypto_tensor.openpangu_v5_7b as _pto  # noqa: E402
        _pto.USE_PTO_FUSED_LAYER = True
        logger.info("* PyPTO fused decode kernel enabled")

    # ---- inject the PyPTO-integrated modeling into the runner's ``models.*`` namespace ----
    from pypto_gym.transformers.openpangu_v5_7b import (  # noqa: E402
        configuration_openpangu_dense,
        modeling_openpangu_dense,
    )
    sys.modules["models.modeling_openpangu_dense"] = modeling_openpangu_dense
    sys.modules["models.configuration_openpangu_dense"] = configuration_openpangu_dense

    # ---- runner flow (mirrors cann-recipes models/pangu-7b/infer.py) ----
    from executor.utils import read_yaml  # noqa: E402
    from executor.utils import data_utils as _du  # noqa: E402
    from models.model_setting import check_vars, update_vars  # noqa: E402
    from runner_openpangu_dense import PanguEmbeddedRunner  # noqa: E402

    yaml_path = args.yaml or os.path.join(os.path.dirname(os.path.abspath(__file__)), "openpangu_v5_7b.yaml")
    runner_settings = read_yaml(yaml_path)
    runner_settings["model_path"] = args.model_path
    if args.max_new_tokens is not None:
        runner_settings.setdefault("data_config", {})["max_new_tokens"] = args.max_new_tokens

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    update_vars(world_size, runner_settings)

    prompt_text = args.prompt

    def _override_prompt(_dataset_dir):
        return [prompt_text]

    _du.generate_default_prompt = _override_prompt

    logger.info("Using device: npu:%s", args.device)
    logger.info("Model path: %s", args.model_path)
    logger.info("PyPTO: %s", "ON" if args.use_pto else "OFF (baseline)")

    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_compile_mode(jit_compile=False)

    preset_prompts, _ = _du.generate_prompt(runner_settings)
    runner = PanguEmbeddedRunner(runner_settings)
    runner.init_model()
    if not args.no_warmup:
        runner.model_generate(preset_prompts, warm_up=True)
        logger.info("Warm up finishes.")
    runner.model_generate(preset_prompts)
    logger.info("model run success")


if __name__ == "__main__":
    main()
