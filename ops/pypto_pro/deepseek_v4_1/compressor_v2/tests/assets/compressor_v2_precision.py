# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AscendC CompressorV2 single-case precision and allocation-guard checks.

Numerical comparison follows ops-transformer compressor_golden.check_result.
All updated regions select tolerances using the operator output dtype.
"""

import logging

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

_PRECISION = {
    "bfloat16": {
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
        "rtol": 0.0078125,
        "atol": 0.0001,
    },
    "float16": {
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
        "rtol": 0.005,
        "atol": 0.000025,
    },
    "float32": {
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
        "rtol": 0.005,
        "atol": 0.000025,
    },
}


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


_ROW_DIVIDER = "-" * 87


_ERROR_DIVIDER = "Error Line" + "-" * 77


_MAX_RE_DIVIDER = "Max-RE line:" + "-" * 75


_LOOP_HEADER = "Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff"


_LOOP_HEADER_FP32 = (
    "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
)


def _format_loop_row(seq, expect_val, real_val, expect_fp32_val=None):
    """格式化单行点位：期望值为 inf/nan 时用字符串列，否则用 7 位小数；expect_fp32 存在时输出双列。"""
    diff_rate = cal_relative_diff_np_isclose(real_val, expect_val)
    if "inf" in str(expect_val) or "nan" in str(expect_val):
        diff_abs = "inf" if "inf" in str(expect_val) else "nan"
        if expect_fp32_val is not None:
            return (
                f"{seq:08d} \t {expect_fp32_val:<7} \t {expect_val:<7} \t "
                f"{real_val:<7} \t {diff_abs:<7} \t {diff_rate:<7}"
            )
        return f"{seq:08d} \t {expect_val:<7} \t {real_val:<7} \t {diff_abs:<7} \t {diff_rate:<7}"
    diff_abs = abs(np.float64(expect_val) - np.float64(real_val))
    if expect_fp32_val is not None:
        return (
            f"{seq:08d} \t {expect_fp32_val:.7f} \t {expect_val:.7f} \t "
            f"{real_val:.7f} \t {diff_abs:.7f} \t {diff_rate:.7f}"
        )
    return f"{seq:08d} \t {expect_val:.7f} \t {real_val:.7f} \t {diff_abs:.7f} \t {diff_rate:.7f}"


def _format_error_row(seq, expect_val, real_val, diff_rate):
    """格式化错误点位明细行（7 位小数定点）。"""
    diff_abs = abs(np.float64(expect_val) - np.float64(real_val))
    return f"{seq:08d} \t {expect_val:.7f} \t {real_val:.7f} \t {diff_abs:.7f} \t {diff_rate:.7f}"


def _display_row_range(real_data, expect_data, start, row_range, expect_fp32_data=None):
    """打印 [start + lo, start + hi) 区间的点位行（绝对序号 = start + idx + 1）。"""
    for idx in row_range:
        seq = start + idx + 1
        fp32 = expect_fp32_data[idx + start] if expect_fp32_data is not None else None
        print_log(
            _format_loop_row(
                seq, expect_data[idx + start], real_data[idx + start], fp32
            )
        )


def print_log(data=None, level="INFO"):
    """Send diagnostics through the standard logger with caller information."""
    severity = getattr(logging, level.upper(), logging.INFO)
    _LOGGER.log(severity, "%s", data, stacklevel=2)


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    """输出每个点位的对比结果（≤20 点位全显示，>20 显示首尾各 10）。"""
    print_log(_ROW_DIVIDER)
    print_log(_LOOP_HEADER_FP32 if expect_fp32_data is not None else _LOOP_HEADER)
    print_log(_ROW_DIVIDER)
    split_count = int(end - start)
    if split_count <= 20:
        _display_row_range(
            real_data, expect_data, start, range(split_count + 1), expect_fp32_data
        )
    else:
        _display_row_range(real_data, expect_data, start, range(10), expect_fp32_data)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        _display_row_range(
            real_data,
            expect_data,
            start,
            range(split_count - 10 + 1, split_count + 1),
            expect_fp32_data,
        )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    """输出错误点位明细 + 最大相对误差点位。"""
    print_log(_ERROR_DIVIDER)
    print_log(_LOOP_HEADER)
    print_log(_ROW_DIVIDER)
    len_err = len(err_idx)
    for count, i in enumerate(err_idx, start=1):
        if count < 10 or (90 < count < 100):
            print_log(
                _format_error_row(
                    i, expect_data[i], real_data[i], relative_diff[count - 1]
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            print_log(
                f"{'...':>8} \t {'...':>7} \t {'...':>7} \t {'...':>7} \t {'...':>7}"
            )
        elif count > 100:
            break

    print_log(_MAX_RE_DIVIDER)
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    for m_count, m_idx in enumerate(m_idx_list, start=1):
        if m_count < 4:
            print_log(
                _format_error_row(
                    m_idx, expect_data[m_idx], real_data[m_idx], max_error
                )
            )
        else:
            break
    print_log(_ROW_DIVIDER)


def _print_verdict(rtol, atol, pct_thd, fulfill_percent, result_str):
    """打印 Rtol/Atol/PctThd/PctRlt/Result 汇总表格。"""
    print_log(_ROW_DIVIDER)
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(_ROW_DIVIDER)
    print_log(
        f"{rtol:.4f}    \t {atol:.6f}  \t {pct_thd:.2f}%   \t {fulfill_percent:.6f}%   \t {result_str}"
    )


def _log_reference_overflows(expected_data):
    """Report non-finite reference values without changing the comparison."""
    infinities = expected_data[np.isinf(expected_data)]
    not_a_numbers = expected_data[np.isnan(expected_data)]
    count = infinities.size + not_a_numbers.size
    if count > 0:
        print_log(
            f"Overflow,size:{count},benchmark_output:{infinities[:10]}, {not_a_numbers[:10]}"
        )


def _float8_storage(values, saturate_infinity):
    """Preserve the upstream FP8 comparison's byte-oriented conversion."""
    values[np.isnan(values)] = 0
    if saturate_infinity:
        values[np.isposinf(values)] = 57344
        values[np.isneginf(values)] = -57344
    return np.frombuffer(values.tobytes(), dtype="uint8")


def _comparison_data(real_data, expected_data):
    """Convert FP8 arrays in the same order as the upstream checker."""
    kind = str(real_data.dtype)
    if kind in ("float8_e4m3fn", "float8_e5m2"):
        saturate_infinity = kind == "float8_e5m2"
        real_data = _float8_storage(real_data, saturate_infinity)
        expected_data = _float8_storage(expected_data, saturate_infinity)
    return real_data, expected_data


def _matching_elements(real_data, expected_data, config):
    """Build the original element-wise tolerance mask."""
    if str(real_data.dtype) == "bfloat16":
        real_data = real_data.astype(np.float32)
        expected_data = expected_data.astype(np.float32)
    return np.isclose(
        real_data,
        expected_data,
        rtol=config["rtol"],
        atol=config["atol"],
        equal_nan=True,
    )


def _relative_errors(real_data, expected_data, error_indices, diff_threshold):
    """Normalize only unmatched errors, retaining the two epsilon additions."""
    if str(expected_data.dtype) == "bool":
        expected_data = expected_data.astype(np.int8)
        real_data = real_data.astype(np.int8)
    absolute_diff = abs(expected_data - real_data)
    magnitude = np.maximum(np.abs(real_data), np.abs(expected_data))
    floor = float((1.0 / (1 << 14)) / diff_threshold)
    denominator = np.add(np.maximum(magnitude, floor), 10e-10)
    epsilon = 10e-10
    relative = absolute_diff / (denominator + epsilon)
    return real_data, expected_data, relative[error_indices]


def _result_verdict(fulfill_percent, relative_errors, pct_threshold, max_diff):
    """Apply the original match-rate and abnormal-error gates."""
    required_percent = (1 - pct_threshold) * 100.0
    verdict = "Pass" if fulfill_percent >= required_percent else "Failed"
    max_error = 0
    if len(relative_errors) > 0:
        # Python max preserves the upstream ordering semantics for NaNs.
        max_error = max(relative_errors[:10_000_000])
        if max_error >= max_diff:
            verdict = "Failed"
    return verdict, required_percent, max_error


def check_result(expect, result, data_type, pct_thd=0.005):
    """Apply the AscendC CompressorV2 comparison criteria without new gates."""
    real_data = result.cpu().numpy().flatten()
    expected_data = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == expected_data.size:
        print_log("Empty actual and reference outputs: comparison Pass")
        return 100.0, "Pass"
    if real_data.size != expected_data.size:
        print_log(
            f"Output size mismatch: actual={real_data.size}, reference={expected_data.size}"
        )
        return 0.0, "Failed"
    _log_reference_overflows(expected_data)
    config = _PRECISION.get(data_type, _PRECISION["float16"])
    end = max(real_data.size - 1, 0)
    count = int(end + 1) if end != 0 else 1
    print_log(f"split_count:{float(count)}; max_diff_hd:{config['max_diff_hd']};")
    real_data, expected_data = _comparison_data(real_data, expected_data)
    matched = _matching_elements(real_data, expected_data, config)
    error_indices = np.where(matched != np.array((True,)))[0]
    real_data, expected_data, relative = _relative_errors(
        real_data, expected_data, error_indices, config["diff_thd"]
    )
    fulfill_percent = float(count - error_indices.size) / float(count) * 100.0
    display_output_np_isclose(real_data, expected_data, 0, end)
    verdict, required_percent, max_error = _result_verdict(
        fulfill_percent, relative, pct_thd, config["max_diff_hd"]
    )
    _print_verdict(
        config["rtol"], config["atol"], required_percent, fulfill_percent, verdict
    )
    if len(relative) > 0:
        print_log(
            f"Max-RelativeError is: {max_error}. Threshold is: {config['max_diff_hd']}."
        )
    if verdict == "Failed":
        display_error_output(
            real_data, expected_data, error_indices, relative[:10_000_000]
        )
    return fulfill_percent, verdict


def check(actual, cache, expected, dtype):
    """Check specified outputs and updated cache with the AscendC criterion."""
    output, mask, expected_cache, updated = expected
    if not (
        actual.shape == output.shape
        and actual.dtype == output.dtype
        and actual.is_contiguous()
    ):
        raise AssertionError("output shape, dtype or layout mismatch")
    actual, cache = actual.cpu(), cache.cpu()
    if cache.shape != expected_cache.shape or cache.dtype != expected_cache.dtype:
        raise AssertionError("cache shape or dtype mismatch")
    dim = cache.shape[-1] // 2
    regions = [("cmp_kv", actual[mask], output[mask])]
    for name, begin in (("kv_state_update", 0), ("score_state_update", dim)):
        region = slice(begin, begin + dim)
        selected = updated[..., region]
        regions.append(
            (name, cache[..., region][selected], expected_cache[..., region][selected])
        )
    # The official caller selects every region's threshold from npu_out.dtype.
    data_type = str(actual.dtype).replace("torch.", "")
    precision = {}
    failures = []
    for name, value, golden in regions:
        percentage, verdict = check_result(
            golden.to(torch.float32), value.to(torch.float32), data_type
        )
        precision[name] = {"fulfill_percent": percentage, "result": verdict}
        if verdict != "Pass":
            failures.append(f"{name}: fulfill_percent={percentage}, result={verdict}")
    if failures:
        raise AssertionError("\n".join(failures))
    unchanged = torch.equal(
        cache[~updated].view(torch.int32), expected_cache[~updated].view(torch.int32)
    )
    if not unchanged:
        raise AssertionError("untouched cache changed")
    return precision


def check_guards(backing, cache):
    """Reject writes to the allocation outside the declared cache view."""
    guard_mask = torch.ones(backing.numel(), dtype=torch.bool)
    block_elements = cache.shape[1] * cache.shape[2]
    for block in range(cache.shape[0]):
        begin = cache.storage_offset() + block * cache.stride(0)
        end = begin + block_elements
        guard_mask[begin:end] = False
    if not torch.all(backing.cpu()[guard_mask] == 7717).item():
        raise AssertionError("allocation guard changed")
