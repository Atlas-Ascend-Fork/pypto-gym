#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The trajectory log: one section per action, candidates as table rows.

This file is for a person. What a machine reads is `search_state.json` and
`selects.jsonl`.
"""

import json
import os

from .config import LOG_NOTE_MARK, NOTE_CHARS
from .state import _fpath, rpath, sdir
from .measurement import objective_text, objectives_of


def append_log(op_dir, op, line):
    """One line into the run's readable record, creating its directory.

    `optimization/` was created by `init` and by nothing else, so the one
    command that runs BEFORE init -- `normalize`, which SKILL.md step 0 mandates
    before anything is measured -- raised FileNotFoundError here on every fresh
    op dir. Its own guard checked `isdir(op_dir)`, the parent, not the directory
    this writes into. A re-run on an op dir that already had `optimization/`
    never saw it, which is why it survived.
    """
    os.makedirs(sdir(op_dir), exist_ok=True)
    with open(rpath(op_dir, op), "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def append_fault(op_dir, rec):
    """One line per environment fault, in the operator's optimization/ folder.

    Kept out of search_state.json on purpose: this is raw device output, it is
    written while a block is still running, and it must survive a process that
    dies before it can save. Append-only JSONL does all three; the state file
    does none of them.

    It exists because the one signal that can end a whole run was the one signal
    nothing kept. A campaign can stop early reporting `env_faults=3` with the
    markers behind those three already discarded three times over -- the
    evaluator dropped the detail, the driver's history was never persisted, and
    the block's `on_trial` returned before recording anything. Afterwards
    there was no way to tell a held device from a mistyped path, which are
    opposite problems.
    """
    os.makedirs(sdir(op_dir), exist_ok=True)
    with open(_fpath(op_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _init_baseline_log(a, opening, objs, pre_vals):
    """Open the trajectory with the baseline every later line is quoted against."""
    append_log(a.op_dir, a.op,
               f"# {a.op} — PANKO optimization\n\n## Baseline\n"
               # EVERY objective, not the first one. The baseline is what the
               # whole run is quoted against, and under two objectives a line
               # that names only the latency cannot say what the front started
               # from -- a reader had to open `search_state.json` to find out
               # what the preopt's workspace was.
               f"- preopt: {preopt_line(objs, pre_vals)}\n"
               f"- P_ref {a.p_ref} us"
               + (" (no reference kernel named; the preopt is the reference and "
                  "there is no target)" if opening["p_ref_via"] == "preopt" else "")
               + f" · j0 {opening['j0']} · device {a.device}")


def _action_label(node):
    """`F-1`, `insert` for an action the optimizer added (it has no catalogue
    id), or `unknown` when the node is not on the frontier to ask."""
    if node is None:
        return "unknown"
    return node.get("catalog_id") or "insert"


def _candidate_note(a, acceptance):
    """The note a candidate's log row carries, when it carries one."""
    # Charged like any other: the device ran the test and the
    # answer was "wrong". Only a candidate that never reached
    # the device is free.
    return ("incorrect (s=0)" if not a.s
            else "undetermined, n unchanged"
            if (acceptance or {}).get("clarity") == "undetermined"
            else "")


def _demand_note(st, pr):
    """The direction an empty frontier can name, as a log fragment."""
    d = pr.get("expansion_demand")
    if not d:
        return ""
    # The one request a single objective cannot make.
    return (f" | DIRECTION: {'lower' if d['direction'] == 'min' else 'raise'} "
            f"`{d['objective']}` past "
            f"{objective_text(objectives_of(st), {d['objective']: d['beyond']})}"
            f" -- the catalogue is spent on it "
            f"({', '.join(d['catalog_spent'])})")


def _in_table(op_dir, op):
    """Is the trajectory file currently inside a candidate table?

    Read back from the file rather than carried in the state. `select` saves
    the state BEFORE it writes the section heading, and every command here is a
    separate process, so a flag set while logging would never reach the next
    one. The file is the only thing both of them see.
    """
    try:
        with open(rpath(op_dir, op), "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    return bool(lines) and lines[-1].lstrip().startswith("|")


def md_cell(text):
    """`text` as one Markdown table cell: no pipes, no line breaks."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _obj_unit(o):
    """The unit a column header names, or "" for a bare number.

    DECLARED, not inferred from where the value is read: `mix_aic` moved from
    `p` to `m.mix_aic` and silently lost its unit and its rounding, which is
    the kind of thing a reader notices only by comparing two logs.
    """
    if o.get("unit"):
        return o["unit"]
    if o["from"].endswith("_b"):
        return "MiB"
    return "us" if o["from"] == "p" else ""


def _obj_cell(o, v):
    """One objective's value, in the unit its column header names."""
    if v is None:
        return "—"
    unit = _obj_unit(o)
    if unit == "MiB":
        return f"{v / 2 ** 20:.1f}"
    if unit == "us":
        return f"{v:.2f}"
    return f"{round(v, 4)}"


def _obj_header(objs):
    """The objective columns, each carrying its unit so the cells stay numeric."""
    return [f"{o['name']}{f' ({_obj_unit(o)})' if _obj_unit(o) else ''}" for o in objs]


def _obj_row(objs, vals):
    return [_obj_cell(o, (vals or {}).get(o["name"])) for o in objs]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def preopt_line(objs, vals):
    """The preopt's value for every objective, in the unit each is read in."""
    return " · ".join(
        f"{o['name']} {_obj_cell(o, (vals or {}).get(o['name']))}"
        + (f" {_obj_unit(o)}" if _obj_unit(o) else "")
        for o in objs)


def log_open_action(op_dir, st, node, line2):
    """Start the section for the action `select` just handed out."""
    delta = " ".join((node.get("delta") or "").split())
    objs = _obj_header(objectives_of(st))
    head = ["eval", "cand"] + objs + ["ratchet", "front", "n", "note"]
    # Right-align the numbers, left-align the words. The separator row is what
    # makes the preceding line a header rather than a paragraph, so the two are
    # always written together.
    align = ["---:", ":---"] + ["---:"] * len(objs) + [":---", ":---", "---:", ":---"]
    append_log(op_dir, st["op"],
               f"\n## {node['id']} · {node.get('catalog_id') or 'insert'} · "
               f"{delta[:78]}{'…' if len(delta) > 78 else ''}"
               f"\n\n{line2}\n\n"
               + _row(head) + "\n" + _row(align))


def log_candidate(op_dir, st, code_hash, row):
    """One measured candidate, as a row of the action's table.

    `row.note` is for something that is true OF THIS CANDIDATE -- that it was
    incorrect, say. It rides in this row rather than in one of its own: a
    separate row for a fact about the row above it is a row spent on nothing,
    and it reads as if something else happened in between. A note about the
    ACTION (a tile block, a close, a lever that could not propose) is a
    different thing and still gets its own row.
    """
    append_log(op_dir, st["op"],
               _row([str(st["progress"]["evals_used"]), f"`{code_hash[:8]}`"]
                    + _obj_row(objectives_of(st), row.vals)
                    + [row.ratchet, row.archive or "—", str(row.n),
                       md_cell(row.note) if row.note else ""]))


def log_note(op_dir, st, text, detail=""):
    """Something that happened inside the current action but is not a candidate.

    A row of the same table, so it keeps its place in the sequence. Before any
    action has been opened there is no table to be a row of, and it is written
    as an ordinary bullet instead.
    """
    body = md_cell(text) + (f" — {md_cell(detail)[:NOTE_CHARS]}" if detail else "")
    if not _in_table(op_dir, st["op"]):
        append_log(op_dir, st["op"], f"- {body}")
        return
    # cand + every objective + ratchet + front + n: the columns between the
    # eval number and the note.
    width = 4 + len(objectives_of(st))
    append_log(op_dir, st["op"],
               _row([LOG_NOTE_MARK] + [""] * width + [body]))


def log_status(op_dir, st):
    """Where the run is, in one line: written at every close and phase change.

    It also ENDS the candidate table -- a status line is a paragraph, and the
    next thing written after it is either an evolve reason or the next action's
    heading, never another row.
    """
    pr, cfg = st["progress"], st["config"]
    budget = cfg.get("eval_budget")
    lim = cfg.get("global_stagnation_limit")
    # Worked, not elapsed -- the figure the limit bounds. The old fallback
    # computed elapsed, which is now a different quantity.
    wall = pr.get("wall_clock_s") or 0
    append_log(op_dir, st["op"],
               f"\n— {pr['evals_used']}"
               + (f"/{budget}" if budget else "") + " evals"
               + f" · best {pr['best_aicore_e2e_us']} us e2e"
               + f" · front {len(st.get('archive') or [])} points"
               + (f" · stagnation {pr['global_stagnation']}/{lim}" if lim else "")
               + f" · wall {wall // 3600}h{(wall % 3600) // 60:02d}m")
