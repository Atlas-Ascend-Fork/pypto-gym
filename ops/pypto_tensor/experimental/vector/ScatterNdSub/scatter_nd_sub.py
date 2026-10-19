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

import logging
import os
import sys
import argparse
import math
import pypto
import torch
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
def scatter_nd_sub_kernel(
    target: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    updates: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32)
):
    for bs_idx, tile_batch in pypto.loop_unroll(0, indices.shape[0], 1, name="LOOP_SCATTER_ND_SUB_L0",
                                                idx_name="bs_idx", unroll_list=[2048, 1024, 512, 256, 1]):
        b_offset = bs_idx
        b_offset_end = bs_idx + tile_batch
        indices_temp = indices[b_offset:b_offset_end, ...]
        pypto.set_vec_tile_shapes(32, 16)
        indices_temp_new = pypto.reshape(indices_temp, [indices_temp.shape[0]])
        indices_tuple = (indices_temp_new, )
        pypto.set_vec_tile_shapes(32, 16)
        neg_updates = pypto.mul(updates[b_offset:b_offset_end], -1)
        accumulate = True
        pypto.index_put_(target, indices_tuple, neg_updates, accumulate)


def test_scatter_nd_sub(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    # Get current device ID (set in main)
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    # 1. prepare pypto data
    input_scenarios = [
        ([10, 16], [1024, 1], [1024, 16]),
        ([1000000, 16], [12900, 1], [12900, 16]),
        ([104007, 8], [2048, 1], [2048, 8]),
        ([22692, 8], [2048, 1], [2048, 8]),
        ([295467, 32], [512, 1], [512, 32]),
        ([3000000, 32], [201892, 1], [201892, 32]),
        ([301044, 32], [512, 1], [512, 32]),
        ([32449, 16], [1024, 1], [1024, 16]),
        ([4635, 8], [2048, 1], [2048, 8]),
        ([6000000, 32], [342312, 1], [342312, 32]),
        ([634054, 32], [512, 1], [512, 32]),
        ([875000, 8], [22704, 1], [22704, 8]),
        ([9153, 16], [1024, 1], [1024, 16]),
        ([934708, 64], [256, 1], [256, 64])
    ]
    target_max_value = 10
    indices_min_value = 0
    updates_max_value = 2

    for idx, (target_shape_list, indices_shape_list, updates_shape_list) in enumerate(input_scenarios):
        logging.info(f"\n=== Generating data for scenario {idx+1} ===")

        target_shape = tuple(target_shape_list)
        indices_shape = tuple(indices_shape_list)
        updates_shape = tuple(updates_shape_list)

        target = torch.rand(target_shape, dtype=torch.float32, device=device) * target_max_value
        target1 = target.clone()

        indices_max_value = min(target_shape[0], target_shape[1])
        indices = torch.randint(
            low=indices_min_value,
            high=indices_max_value,
            size=indices_shape,
            dtype=torch.int32,
            device=device
        )

        updates = torch.rand(updates_shape, dtype=torch.float32, device=device) * updates_max_value
        scatter_nd_sub_kernel(target, indices, updates)

        # 2. prepare tensorflow data
        target_tf = tf.convert_to_tensor(target1.cpu().numpy())
        indices_tf = tf.convert_to_tensor(indices.cpu().numpy())
        updates_tf = tf.convert_to_tensor(updates.cpu().numpy())

        target_var = tf.compat.v1.Variable(target_tf)
        indices_var = tf.compat.v1.Variable(indices_tf)
        updates_var = tf.compat.v1.Variable(updates_tf)
        tf_output = tf.compat.v1.scatter_nd_sub(target_var, indices_var, updates_var)

        # 3. compare pypto vs tensorflow output
        max_diff = np.abs(target.cpu().numpy() - tf_output.numpy()).max().item()
        if run_mode == "npu":
            logging.info(f"Max difference from Tensorflow: {max_diff:.6f}")
            assert max_diff < 1e-1, "Result mismatch!"
        logging.info("✓ Combined operations completed successfully")
    logging.info("finished")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO scatter_nd_sub Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scatter_nd_sub::test_scatter_nd_sub
            Run the scatter_nd_sub::test_scatter_nd_sub example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1). If not specified, the example will run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )

    args = parser.parse_args()

    # Define available examples
    examples = {
        "scatter_nd_sub::test_scatter_nd_sub": {
            'name': 'scatter_nd_sub',
            'description': 'scatter_nd_sub implementation',
            'function': test_scatter_nd_sub
        }
    }

    # List examples if requested
    if args.list:
        logging.info("Available Examples")
        logging.info("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            logging.info(f"  ID: {ex_id}")
            logging.info(f"     name: {ex_info['name']}")
            logging.info(f"     description: {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            logging.error(f"ERROR: Invalid example ID: {args.example_id}")
            logging.info(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            logging.info("\nUse --list to see all available examples.")
            raise RuntimeError("Test execution failed")

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO test_scatter_nd_sub Example")
    logging.info("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        # Run all examples
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        import torch_npu
        torch.npu.set_device(device_id)
        logging.info("Running examples that require NPU hardware...")
        logging.info("(Make sure CANN environment is configured and NPU is available)\n")

    try:
        for ex_id, ex_info in examples_to_run:
            logging.info(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            logging.info("=" * 60)
            logging.info("All scatter_nd_sub tests passed!")
            logging.info("=" * 60)

    except Exception as e:
        logging.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
