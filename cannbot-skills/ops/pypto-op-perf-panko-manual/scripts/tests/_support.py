# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared test support: the harness under test, the paths to it, an envelope for
a run with no device, and a writer for synthetic platform inis.

`init` DERIVES the chip envelope from the platform ini and refuses when it
cannot, rather than falling back to a transcribed default -- so a test running
off-device has to supply one, exactly as a user on a box without CANN would.
`CHIP_ENVELOPE_OFFLINE` is that supplied envelope. It describes no device: it is
a set of round numbers large enough for the toy kernels these tests compile.

`write_platform_ini` writes the ini SHAPE -- the sections and option names
`chip_profile` reads -- with values chosen for the test. No real SKU's figures
are stored in this repository, and no ini ships as a file: a test that needs one
writes it into its own temporary directory.
"""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

HARNESS = SCRIPTS_DIR / "panko_harness.py"
CATALOG = SCRIPTS_DIR.parent / "references" / "action_catalog.json"

# The harness loaded once, by path, so a test can call the internals it exists
# to test. Every test file that needs it imports this one rather than repeating
# the eight lines of spec/module/exec_module boilerplate.
# Every test runs against a private PANKO_HOME: the outcome store and the
# learned prior table are machine-wide by design, and a test must neither
# write into the developer's nor read what another test left there.
os.environ.setdefault("PANKO_HOME", tempfile.mkdtemp(prefix="panko-home-test-"))


# Lowest layer first, so a name defined in one layer and re-exported by a higher
# one ends up pointing at the higher one's view of it, which is the one a case
# calling `harness.<name>` means. One list, because the namespace and the source
# text are two views of the same package and a layer added to one of them that
# is missing from the other is a silent gap in what the suite sees.
PANKO_LAYERS = ("config", "state", "measurement", "front", "stopping", "log",
                "selection", "delivery", "init", "block", "lifecycle")


def _under_test():
    """ONE namespace over the whole package, for the tests alone.

    The package is layered and a module names the layer it reads from; a case
    asserting about `close` should not also have to assert which of the eleven
    modules the helper it calls lives in, because that is the thing the
    layering is free to change. So this flattens them and the cases say
    `harness.<name>`. It is a test convenience and nothing in `scripts/` may
    do it: it is exactly the single mutable namespace the package was
    restructured to remove. A case that needs to PATCH a name patches the
    module that defines it (`selection.tile_bayes_propose`), not this view.

    Imported here rather than at the top of the file, and not by accident:
    `SCRIPTS_DIR` has to reach `sys.path` before the package can be found at
    all, and `PANKO_HOME` has to be the private one before any module reads
    it. A function is how that order is stated rather than merely relied on.
    """
    import panko_harness  # noqa: PLC0415

    namespace = types.ModuleType("panko_under_test")
    for module in [importlib.import_module(f"panko.{m}") for m in PANKO_LAYERS] + [
        panko_harness
    ]:
        for name, value in vars(module).items():
            if not name.startswith("__"):
                setattr(namespace, name, value)
    return namespace


harness = _under_test()

CHIP_ENVELOPE_OFFLINE = {
    "ub_kb": 192,
    "l1_kb": 512,
    "l0a_kb": 64,
    "l0b_kb": 64,
    "l0c_kb": 128,
    "cube_cores": 20,
    "vector_cores": 40,
    # The hypervolume's reference on a byte-valued objective. Optional in a
    # supplied envelope, and supplied HERE on purpose: without it
    # `hv_reference_point` returns None and every CLI test would exercise the
    # degraded path (axis records only) while looking like it covered the
    # hypervolume. `test_hv_reference.py` covers the degraded path explicitly.
    "memory_mib": 65536,
}

# Append to any `init` argv in a test.
CHIP_ARGS = ["--chip-envelope", json.dumps(CHIP_ENVELOPE_OFFLINE)]

# The SHIPPED default is one objective (`aicore_e2e`) -- that is the
# single-objective mode, and `test_default_mode.py` pins it. This suite is about
# the multi-objective mode, so every `init` here DECLARES the second objective
# instead of inheriting it: `run_cli` adds `objectives: [aicore_e2e, workspace]`
# to an init that names none (see `_with_suite_objectives`). A test that wants a
# different set -- one objective, three, `mix_aic` -- passes its own `--config`
# and that is left alone.
#
# `init` refuses a preopt that is missing a declared objective's value (see
# `_refuse_objectives`), so every init in the suite carries the workspace too.
# Tests about the workspace itself pass their own value after this.
SUITE_OBJECTIVES = ["aicore_e2e", "workspace"]

INIT_ARGS = CHIP_ARGS + ["--preopt-workspace-b", str(1 << 20)]


# Every objective in `config.objectives` must have a value on every measured
# candidate, or the run halts (`_refuse_objectives`). The suite's tests about
# other things are not about that plumbing, so a `record` / `init` that does not
# mention them gets them here; the MO tests pass their own.
DEFAULT_RECORD_WORKSPACE = ["--workspace-b", str(1 << 20)]

# `p` IS `aicore_e2e` now, so what a fixture has to supply is the OTHER
# objective: `mix_aic`. It has to MOVE WITH `p` -- a constant would make every
# candidate tie on that axis and quietly change what the dominance tests are
# testing -- and 1.0 keeps every J in this suite arithmetically what it always
# was, so a test that fails here fails for a reason and not for a change of
# scale. The tests that are ABOUT the two disagreeing pass their own values.
MIX_AIC_OF_P = 1.0

# The learned `prior_gain` reads a table of historical kept-rates, and NO table
# ships: a user's prior is built from their own campaigns (`prior_store`). The
# suite still has to exercise the learned path, so it writes a SYNTHETIC table
# once and pins every init to it, the same way `write_platform_ini` writes an
# ini shape rather than shipping a real SKU's. The numbers are chosen to make a
# posterior that moves; they are nobody's measurements.
SUITE_PRIOR_TABLE = {
    "meta": {"synthetic": "written by the test suite; not a measurement"},
    "global": {
        "n_meas": 2000,
        "n_kept": 200,
        "n_s0": 400,
        "sum_log_gain_kept": 10.0,
        "n_kept_with_gain": 200,
        "n_kept_without_gain": 0,
        "n_campaigns": 40,
    },
    "by_catalog": dict(
        {
            f"C-{i}": {
                "n_meas": 100,
                "n_kept": 4 + 2 * i,
                "n_s0": 20,
                "sum_log_gain_kept": 0.5 + 0.1 * i,
                "n_kept_with_gain": 4 + 2 * i,
                "n_kept_without_gain": 0,
                "n_campaigns": 8,
            }
            for i in range(8)
        },
        **{
            "insert": {
                "n_meas": 250,
                "n_kept": 45,
                "n_s0": 50,
                "sum_log_gain_kept": 2.0,
                "n_kept_with_gain": 45,
                "n_kept_without_gain": 0,
                "n_campaigns": 20,
            }
        },
    ),
    "by_catalog_predicate": {},
}


def write_prior_table(path, table=None):
    """Write a synthetic prior table and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table or SUITE_PRIOR_TABLE), encoding="utf-8")
    return str(path)


_SUITE_TABLE_PATH = write_prior_table(
    Path(tempfile.mkdtemp(prefix="panko-prior-test-")) / "action_prior_table.json"
)


def _flag_value(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _with_mix_aic(args, p_flag, metrics_flag):
    """Give the call a `mix_aic` unless it already carries one."""
    carried = _flag_value(args, metrics_flag) or ""
    if "mix_aic" in carried or metrics_flag.replace("metrics", "measurement") in args:
        return args
    p = _flag_value(args, p_flag)
    try:
        mix = round(float(p) * MIX_AIC_OF_P, 6)
    except (TypeError, ValueError):
        return args
    if carried:
        try:
            d = json.loads(carried)
        except ValueError:
            return args
        d["mix_aic"] = mix
        out = list(args)
        out[out.index(carried)] = json.dumps(d)
        return tuple(out)
    return tuple(args) + (metrics_flag, json.dumps({"mix_aic": mix}))


def _with_suite_objectives(args):
    """Declare the suite's two objectives on an `init` that names none.

    The default is single-objective, so an init that says nothing would run the
    single-objective search and the front tests would be testing the wrong mode.
    A `--config` that already carries `objectives` is the test's own choice and
    is returned untouched; one that carries other keys gets the objectives
    merged in, because it is not a statement about the objectives.
    """
    carried = _flag_value(args, "--config")
    if carried is None:
        return _with_hv_ceiling(
            tuple(args)
            + (
                "--config",
                json.dumps(
                    {
                        "objectives": list(SUITE_OBJECTIVES),
                        "action_prior_table": _SUITE_TABLE_PATH,
                    }
                ),
            )
        )
    try:
        cfg = json.loads(carried)
    except ValueError:
        return args
    if not isinstance(cfg, dict) or "objectives" in cfg:
        return args
    cfg["objectives"] = list(SUITE_OBJECTIVES)
    cfg.setdefault("action_prior_table", _SUITE_TABLE_PATH)
    out = list(args)
    if carried not in out:  # `--config=<json>`: left to the test
        return args
    out[out.index(carried)] = json.dumps(cfg)
    return _with_hv_ceiling(tuple(out))


def _with_hv_ceiling(args):
    """A byte objective needs a ceiling, so an init we just made multi-objective
    needs one too.

    Only when the suite added the objectives and the test named neither an
    envelope nor a ceiling -- a test that supplies its own says what it is
    about, and `test_hv_reference.py` is about running without one.
    """
    if any(a in args for a in ("--chip-envelope", "--chip-memory-mib")):
        return args
    return tuple(args) + ("--chip-memory-mib", "65536")


def _with_preopt_workspace(args):
    """Give a suite `init` the preopt workspace its declared objectives need.

    `_with_suite_objectives` declares `workspace`, and `init` refuses a preopt
    that is missing a declared objective's value. Tests that build their own
    argv -- the chip-resolution tests, which are about the envelope and not
    about the objectives -- would otherwise be refused for a reading they never
    meant to be about. A test that passes its own workspace keeps it.
    """
    declared = ("--preopt-workspace-b", "--preopt-workspace-state",
                "--preopt-measurement")
    if any(a in args for a in declared):
        return args
    return tuple(args) + ("--preopt-workspace-b", str(1 << 20))


def _with_objective_values(args):
    if not args:
        return args
    if args[0] == "init":
        return _with_preopt_workspace(
            _with_mix_aic(
                _with_suite_objectives(args), "--preopt-p", "--preopt-metrics"
            )
        )
    if args[0] != "record":
        return args
    if "--unevaluated" not in args and "--s" in args:
        if _flag_value(args, "--s") != "0":
            args = _with_mix_aic(args, "--p", "--metrics")
    if any(a in args for a in ("--workspace-b", "--workspace-state", "--unevaluated")):
        return args
    return tuple(args) + tuple(DEFAULT_RECORD_WORKSPACE)


def write_platform_ini(
    directory,
    soc,
    *,
    cube=4,
    vector=8,
    ub=32768,
    l0a=8192,
    l0b=8192,
    l0c=16384,
    l1=65536,
    arch="0000",
    short=None,
    vector_ub=None,
):
    """Write `<soc>.ini` into `directory` and return its path.

    The sections and option names are the ones `chip_profile.FIELDS` and
    `COUNTS` read; the numbers are the test's, not any device's. Sizes are
    written in BYTES, which is what an ini holds and what `read_ini` converts.

    `vector_ub` adds a second `[VectorCoreSpec] ub_size`, which some platforms
    carry and others do not -- the case that makes a flat key scan ambiguous.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = [
        "[SoCInfo]",
        f"ai_core_cnt={cube}",
        f"cube_core_cnt={cube}",
        f"vector_core_cnt={vector}",
        "",
        "[AICoreSpec]",
        f"ub_size={ub}",
        f"l0_a_size={l0a}",
        f"l0_b_size={l0b}",
        f"l0_c_size={l0c}",
        f"l1_size={l1}",
        "",
    ]
    if vector_ub is not None:
        body += ["[VectorCoreSpec]", f"ub_size={vector_ub}", ""]
    body += [
        "[version]",
        f"SoC_version={soc}",
        f"Short_SoC_version={short or soc}",
        f"NpuArch={arch}",
        "",
    ]
    path = directory / f"{soc}.ini"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


# The smallest program the harness accepts: one jit function with one tile
# directive. Nearly every CLI test needs a kernel and does not care what it
# computes, and a copy of it in each file is a copy of the shape the harness
# requires -- a `pypto.frontend.jit` decorator it can find and a tile call it
# can move. A test about a DIFFERENT shape (two directives, a cube tile, a
# factory) writes its own, because the shape is then what the test is about.
DEMO_KERNEL = """import pypto


@pypto.frontend.jit()
def demo(a, b):
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def init_argv(op_dir, op_file, *extra, catalog=CATALOG):
    """The standard `init` argv of a demo campaign: the reference numbers every
    test shares, `--catalog` included unless catalog=None (the path-resolution
    tests omit it on purpose, because the default is the thing under test).
    Extra flags go last.
    """
    argv = [
        "init",
        "--op-dir",
        str(op_dir),
        "--op",
        "demo",
        "--p-ref",
        "100",
        "--preopt-p",
        "500",
    ]
    if catalog is not None:
        argv += ["--catalog", str(catalog)]
    argv += ["--allow-unversioned", "--op-file", str(op_file)]
    return argv + list(extra)


def record_argv(op_dir, digest, p):
    """The `record` argv of one measured candidate of the demo campaign."""
    return [
        "record",
        "--op-dir",
        str(op_dir),
        "--parent",
        "u1",
        "--code-hash",
        digest,
        "--s",
        "1",
        "--p",
        p,
    ]


def feasible_argv(op_dir, op_file):
    """The `feasible` argv of one candidate of the demo campaign."""
    return [
        "feasible",
        "--op-dir",
        str(op_dir),
        "--parent",
        "u1",
        "--op-file",
        str(op_file),
    ]


def run_cli(*args, cwd=None, env=None):
    """Run the harness CLI and return (payload, exit code).

    Every CLI test needs exactly this, and a copy of it in each file is a copy of
    the contract: one JSON object on stdout, and the exit code carries the
    refusal. `cwd` and `env` set what the run can see; a test that has to invoke
    a DIFFERENT copy of the script -- the path-resolution tests run the installed
    symlink -- builds its own argv, because which script runs is the thing those
    tests are about.
    """
    if env is None:
        env = dict(os.environ)
    args = _with_objective_values(args)
    r = subprocess.run(
        [sys.executable, str(HARNESS), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout), r.returncode


def file_hash(path):
    """The code hash the harness computes for a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# A kernel with BOTH tile directives: one cube call carrying three operand
# shapes and one vec call. Every test about the derived domain, the capacity
# bound or tile repair needs two kinds of site in one file, and the shape is
# the fixture rather than anything a single test chose.
CUBE_KERNEL = """import pypto


@pypto.frontend.jit()
def k(a, b):
    pypto.set_cube_tile_shapes([128, 256], [64, 128], [128, 256])
    pypto.set_vec_tile_shapes(64, 128)
    return a + b
"""


def harness_source():
    """The CLI and every module of the package, as one text.

    A case that asserts about the CODE should not also assert which file holds
    it: these are one program, layered, and a line that moved between the
    layers has not changed. Reading them together also keeps an `assertNotIn`
    honest -- it would otherwise start passing the moment its subject moved to
    a sibling module.
    """
    files = [SCRIPTS_DIR / "panko" / f"{m}.py" for m in PANKO_LAYERS]
    return "\n".join(
        f.read_text(encoding="utf-8")
        for f in files + [SCRIPTS_DIR / "panko_harness.py"]
    )


def record_measurement(case, src, p, ws, parent="u1"):
    """Write `src` to the case's kernel file and record one measurement of it.

    The five lines this replaces -- write the source, hash it, hand the hash to
    `record` -- are what every frontier test does between one measurement and
    the next, and a copy of them in each file is a copy of the contract that
    the recorded hash is the hash of the file that was measured. Returns
    (code hash, the command's reply).
    """
    case.f.write_text(src, encoding="utf-8")
    code_hash = file_hash(case.f)
    reply, _ = run_cli(
        "record",
        "--op-dir",
        str(case.op_dir),
        "--parent",
        parent,
        "--code-hash",
        code_hash,
        "--s",
        "1",
        "--p",
        str(p),
        "--workspace-b",
        str(ws),
    )
    return code_hash, reply


def new_op_dir(case, kernel, op="demo"):
    """A temporary operator directory holding `kernel`, cleaned up with `case`.

    Every CLI test starts from the same six lines -- a temp dir, an operator
    directory under it, the kernel written into `<op>_impl.py` -- and a copy of
    them in each file is a copy of the layout the harness expects. Sets
    `case.tmp_dir`, `case.op_dir` and `case.pre` (the code hash) and returns the
    kernel's path; the caller names it, because the tests disagree about what to
    call it and the name is in their assertions.
    """
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    case.tmp_dir = tmp
    case.op_dir = Path(tmp.name) / op
    case.op_dir.mkdir(parents=True)
    path = case.op_dir / f"{op}_impl.py"
    path.write_text(kernel, encoding="utf-8")
    case.pre = file_hash(path)
    return path


def read_state(op_dir):
    """The search state the harness wrote under `op_dir`."""
    return json.loads(
        (Path(op_dir) / "optimization" / "search_state.json").read_text(
            encoding="utf-8"
        )
    )
