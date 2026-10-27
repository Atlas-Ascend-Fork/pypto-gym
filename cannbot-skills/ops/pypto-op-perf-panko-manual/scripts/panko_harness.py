#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""panko_harness.py -- the CLI for the PANKO deterministic control core.

One subcommand per transition, one JSON object on stdout, and the exit code
carries the refusal. Everything it dispatches to lives in the `panko` package
beside this file, which is layered so that it can be read in one order:
`config` imports nothing from the package, and each module above it uses only
the ones below.

This file is the top of that stack. Nothing in the package imports it, which is
what lets the package be imported by anything else -- a test, another tool --
without this script running as `__main__` a second time.
"""

import argparse
import sys

from panko.config import DEFAULT_CATALOG, Refusal, WORKSPACE_STATES
from panko.delivery import cmd_pareto, cmd_report, cmd_verify
from panko.init import cmd_init, cmd_normalize
from panko.block import cmd_block
from panko.lifecycle import cmd_close, cmd_evolve, cmd_feasible, cmd_probe, cmd_record, cmd_select, cmd_stop


# --------------------------------------------------------------------------- cli
def _parser_block(sub):
    """Arguments for `block`."""
    q = sub.add_parser("block")
    q.set_defaults(fn=cmd_block)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", required=True, help="the action this block belongs to")
    q.add_argument("--test-command", dest="test_command", required=True,
                   help="shell string that exits 0 iff golden+layout pass")
    q.add_argument("--reason", default="",
                   help="why tuning is needed now; recorded for audit")
    q.add_argument("--live-tiles", dest="live_tiles", type=int, default=0,
                   help="how many tile-shaped tensors are simultaneously resident. "
                        "A FACT about the program, supplied because the extractor "
                        "cannot read it after a fusion or a flatten. Never a bound.")
    q.add_argument("--max-trials", dest="max_trials", type=int, default=0,
                   help="upper bound requested by the optimizer; the core clamps it "
                        "against the remaining budget and the run-level share")
    q.add_argument("--lint-root", dest="lint_root", default="")
    q.add_argument("--evaluation-mode", choices=("subprocess", "resident", "cached", "prepared"),
                   default="subprocess")


def _parser_normalize(sub):
    """Arguments for `normalize`."""
    q = sub.add_parser("normalize")
    q.set_defaults(fn=cmd_normalize)
    q.add_argument("--before", required=True,
                   help="the kernel as generated, before normalisation")
    q.add_argument("--after", required=True,
                   help="the normalised kernel")
    q.add_argument("--op-dir", default="", help="optional, to log the verdict")
    q.add_argument("--op", default="")


_PREOPT_METRIC_ARGS = (
    ("--preopt-util", "preopt_util",
     "preopt core-util %% (optional; enables first-move symptom bias)"),
    ("--preopt-bubble", "preopt_bubble",
     "preopt bubble %% (optional; enables first-move symptom bias)"),
    ("--preopt-aic-util", "preopt_aic_util",
     "preopt cube-pipe core-util %% (optional; enables the per-pipe symptoms at INIT)"),
    ("--preopt-aiv-util", "preopt_aiv_util",
     "preopt vector-pipe core-util %% (optional)"),
    ("--preopt-pred-stall", "preopt_pred_stall",
     "preopt predecessor-wait %% (optional)"),
    ("--preopt-aic-bubble", "preopt_aic_bubble",
     "preopt cube-pipe wait-schedule %% (optional)"),
    ("--preopt-aiv-bubble", "preopt_aiv_bubble",
     "preopt vector-pipe wait-schedule %% (optional)"),
)


def _parser_init(sub):
    """Arguments for `init`."""
    q = sub.add_parser("init")
    q.set_defaults(fn=cmd_init)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--op", required=True)
    # OPTIONAL. Without it the preopt is the reference and there is no target
    # to reach -- see `_init_p_ref`. It used to be required, so a campaign
    # with no reference kernel had to invent a number, and one did: a run was
    # given a reference its own preopt already beat.
    q.add_argument("--p-ref", type=float, default=None)
    q.add_argument("--device", default="")
    # Defaults to this skill's own catalogue, so a caller in any working
    # directory gets the right one without being told where it is.
    # Last resort, for a box where the platform ini is unreachable. Recorded as
    # explicit:cli so a report never reads as derived when it was supplied.
    q.add_argument("--chip-envelope", dest="chip_envelope", default="")
    # The memory this kernel may hold, in MiB: the hypervolume's reference point
    # on a byte-valued objective. Defaults to the chip's whole memory from the
    # platform ini, which assumes the kernel runs alone.
    q.add_argument("--chip-memory-mib", dest="chip_memory_mib", default=None)
    # Which case of a factory kernel's per-case config table to tune.
    q.add_argument("--config-key", dest="config_key", default=None)
    q.add_argument("--catalog", default=DEFAULT_CATALOG)
    q.add_argument("--config", default="")
    q.add_argument("--preopt-s", dest="preopt_s", type=int, default=1)
    q.add_argument("--preopt-p", dest="preopt_p", type=float, required=True)
    _parser_init_preopt(q)
    for flag, dest, help_text in _PREOPT_METRIC_ARGS:
        q.add_argument(flag, dest=dest, type=float, default=None, help=help_text)
    q.add_argument("--op-file", dest="op_file", default="",
                   help="working <op>_impl.py the harness snapshots/restores (default: <op-dir>/<op>_impl.py)")
    q.add_argument("--lint-root", dest="lint_root", default="",
                   help="the same --lint-root the blocks will be given; checked here so a "
                        "wrong one is reported before it faults every trial")
    q.add_argument("--allow-unversioned", dest="allow_unversioned", action="store_true",
                   help="start even when the harness build cannot be identified (CI only; "
                        "a run started this way must not be reported)")


def _parser_select(sub):
    """Arguments for `select`."""
    q = sub.add_parser("select")
    q.set_defaults(fn=cmd_select)
    q.add_argument("--op-dir", required=True)


def _parser_record(sub):
    """Arguments for `record`."""
    q = sub.add_parser("record")
    q.set_defaults(fn=cmd_record)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", required=True)
    q.add_argument("--code-hash", dest="code_hash", required=True)
    q.add_argument("--s", type=int, required=True)
    q.add_argument("--p", type=float, default=0.0)
    # None, NOT 0.0. A reading that was never taken and a kernel that genuinely
    # idled are different facts, and defaulting to zero made them the same one:
    # `_symptom_reboost` gates on `util > 0 or bubble > 0`, so every candidate
    # recorded without a reading silently disabled the device half of the
    # symptom index. In one campaign that was every block winner, spine nodes
    # included, and no device symptom fired for the whole run.
    q.add_argument("--util", type=float, default=None)
    q.add_argument("--bubble", type=float, default=None)
    # Per-pipe, from measure_latency (`aic_util` / `aiv_util` / `pred_stall` in
    # its result dict). Optional: they exist only when a bubble_analysis.log was
    # written, and the installed draw_swim_lane.py often skips it.
    q.add_argument("--aic-util", dest="aic_util", type=float, default=None,
                   help="cube-pipe core-util %% (optional; enables the per-pipe symptoms)")
    q.add_argument("--aiv-util", dest="aiv_util", type=float, default=None,
                   help="vector-pipe core-util %% (optional)")
    q.add_argument("--pred-stall", dest="pred_stall", type=float, default=None,
                   help="predecessor-wait %% (optional; separates a serial recurrence from a full pipe)")
    q.add_argument("--aic-bubble", dest="aic_bubble", type=float, default=None,
                   help="cube-pipe wait-schedule %% (optional)")
    q.add_argument("--aiv-bubble", dest="aiv_bubble", type=float, default=None,
                   help="vector-pipe wait-schedule %% (optional; rises when tasks get too big to fill the cores)")
    # The structural path runs the measurement under `timeout <eval_timeout_s>`
    # itself (SKILL.md), so only the caller can tell the harness that nothing
    # was measured. Without this the timeout arrives as `--s 0`, which is a
    # verdict about the candidate, and is charged as an evaluation.
    # Re-measurement of a program the archive already holds: more readings for
    # the SAME bytes, to settle a dominance the interval left undetermined.
    # Not charged, because `eval_budget` counts distinct programs measured and
    # this is not one; and it must bypass the duplicate rejection, which exists
    # to stop a resubmission being charged rather than to stop readings.
    q.add_argument("--remeasure", action="store_true",
                   help="add readings to an archive point already measured: not "
                        "charged, does not touch the ratchet, re-derives the front")
    q.add_argument("--unevaluated", default=None,
                   choices=["compile_timeout", "oom", "device_unavailable"],
                   help="the candidate was NOT measured: not charged to "
                        "eval_budget and not recorded as s=0")
    # measure_latency's `workspace_b` / `workspace_state`. Diagnostic in a
    # single-objective run; `unavailable` must be passed as such, never as 0.
    q.add_argument("--workspace-b", dest="workspace_b", type=int, default=None)
    # The whole `measure_latency` result, so `m` cannot be under-filled by a
    # caller who passed only the flags they remembered. See `cli_measurement`.
    q.add_argument("--measurement", default=None,
                   help="measure_latency's result dict as JSON: every reading "
                        "in it is carried on `m`, which is where an objective "
                        "reads its value from. Prefer this over spelling the "
                        "readings out one flag at a time")
    q.add_argument("--metrics", default=None,
                   help='any further readings as JSON, merged into `m`: '
                        '\'{"dm_bytes": 12345}\'. A user-declared objective reads '
                        'its value from here (config.objectives `from: m.<key>`)')
    q.add_argument("--workspace-state", dest="workspace_state",
                   choices=WORKSPACE_STATES, default=None)
    # The individual warm readings behind `--p` (measure_latency `samples`).
    # Optional: without them the candidate is recorded exactly as before, and
    # only the noise estimate goes without this measurement.
    q.add_argument("--samples", default=None,
                   help="warm readings behind --p, comma-separated (optional; "
                        "feeds the run's noise estimate)")


def _parser_feasible(sub):
    """Arguments for `feasible`."""
    q = sub.add_parser("feasible")
    q.set_defaults(fn=cmd_feasible)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--parent", default="")
    q.add_argument("--op-file", dest="op_file", default="")
    # 0 = use the run's chip envelope. The old literals were a second, stale
    # copy of it: --l1-kb defaulted to 192 against a 512 KB buffer.
    q.add_argument("--ub-kb", dest="ub_kb", type=int, default=0)
    q.add_argument("--l1-kb", dest="l1_kb", type=int, default=0)
    q.add_argument("--mL1", dest="m_l1", type=int, default=None)
    q.add_argument("--kL1", dest="k_l1", type=int, default=None)


def _parser_probe(sub):
    """Arguments for `probe`."""
    q = sub.add_parser("probe")
    q.set_defaults(fn=cmd_probe)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--code-hash", dest="code_hash", required=True)


def _parser_close(sub):
    """Arguments for `close`."""
    q = sub.add_parser("close")
    q.set_defaults(fn=cmd_close)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--action", required=True)


def _parser_evolve(sub):
    """Arguments for `evolve`."""
    q = sub.add_parser("evolve")
    q.set_defaults(fn=cmd_evolve)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--edits", required=True)


def _parser_stop(sub):
    """Arguments for `stop`."""
    q = sub.add_parser("stop")
    q.set_defaults(fn=cmd_stop)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--force", default="",
                   help="record this stop_reason when no condition fired, e.g. session_ended. "
                        "For the campaign driver to call after the agent exits; the agent must "
                        "never pass it.")


def _parser_report(sub):
    """Arguments for `report`."""
    q = sub.add_parser("report")
    q.set_defaults(fn=cmd_report)
    q.add_argument("--op-dir", required=True)


def main():
    p = argparse.ArgumentParser(description="Deterministic PANKO control core.")
    sub = p.add_subparsers(dest="cmd", required=True)

    _parser_block(sub)
    _parser_normalize(sub)
    _parser_init(sub)
    _parser_select(sub)
    _parser_record(sub)
    _parser_feasible(sub)
    _parser_probe(sub)
    _parser_close(sub)
    _parser_evolve(sub)
    _parser_stop(sub)
    _parser_verify(sub)
    _parser_pareto(sub)
    _parser_report(sub)

    a = p.parse_args()
    try:
        a.fn(a)
    except Refusal as refusal:
        sys.exit(refusal.code)


def _parser_init_preopt(q):
    """The preopt measurement's own readings, as `init` takes them."""
    # `compile_s` from the preopt's own measure_latency run. `eval_timeout_s` is
    # derived from it, because the right value is a property of the operator:
    # compile time rises with the workspace a kernel allocates, and candidates
    # grow it from there. Optional -- without it the configured default
    # stands and `config.eval_timeout_source` records that it was not derived.
    q.add_argument("--preopt-compile-s", dest="preopt_compile_s", type=float,
                   default=None,
                   help="the preopt measurement's compile_s; eval_timeout_s is "
                        "derived from it (optional)")
    q.add_argument("--preopt-hash", dest="preopt_hash", default="")
    # The preopt's whole `measure_latency` result, for the same reason `record`
    # takes one: `aicore_e2e` is an objective and must not depend on the caller
    # remembering a flag. See `cli_measurement`.
    q.add_argument("--preopt-measurement", dest="preopt_measurement", default=None,
                   help="measure_latency's result dict for the preopt, as JSON")
    q.add_argument("--preopt-metrics", dest="preopt_metrics", default=None,
                   help="the preopt's further readings as JSON, merged into `m` "
                        "(see `record --metrics`)")
    q.add_argument("--preopt-workspace-b", dest="preopt_workspace_b", type=int, default=None,
                   help="the preopt measurement's workspace_b (measure_latency)")
    q.add_argument("--preopt-workspace-state", dest="preopt_workspace_state",
                   choices=WORKSPACE_STATES, default=None)
    q.add_argument("--preopt-samples", dest="preopt_samples", default=None,
                   help="the preopt measurement's warm readings, comma-separated "
                        "(measure_latency `samples`); seeds the run's noise estimate")


def _parser_pareto(sub):
    """Arguments for `pareto`."""
    q = sub.add_parser("pareto")
    q.set_defaults(fn=cmd_pareto)
    q.add_argument("--op-dir", required=True)


def _parser_verify(sub):
    """Arguments for `verify`."""
    q = sub.add_parser("verify")
    q.set_defaults(fn=cmd_verify)
    q.add_argument("--op-dir", required=True)
    q.add_argument("--samples", default=None,
                   help="the staged arm's warm readings, comma-separated; omit to "
                        "stage the next arm")
    q.add_argument("--skip", default="",
                   help="close the verification without measuring (e.g. "
                        "device_unavailable); the report then says UNVERIFIED")


if __name__ == "__main__":
    main()
