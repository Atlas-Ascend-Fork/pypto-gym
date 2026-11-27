# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare structural candidates using resolved tile shapes and declared slot addresses."""

__all__ = ["canonical", "same_program"]

import ast
import logging

from memory.onchip_layout import author_tile_bytes
from source_analysis import tile_sites

logger = logging.getLogger(__name__)


def _canonical_addrs(site, addresses):
    """Use a scalar base only when it represents the full declared slot sequence."""
    base = addresses[0]
    if len(addresses) == 1:
        return str(base)
    stride = author_tile_bytes(site)
    if stride and all(addr == base + i * stride for i, addr in enumerate(addresses)):
        return str(base)
    return str(addresses)


def canonical(src, extra_consts=None):
    """Return the canonical AST and resolved entries. Preserve unresolved expressions."""
    from . import apply as apply_mod

    sites = tile_sites.read_sites(src, apply_mod.source_constants(extra_consts))
    edits, resolved = [], []

    for site in sites:
        for axis, entry in enumerate(site.shape):
            if entry.value is None or entry.text == str(entry.value):
                continue
            what = "%s shape[%d]" % (site.site_id, axis)
            edits.append(apply_mod.Edit(entry.start, entry.end, str(entry.value), what))
            resolved.append("%s %s=%d" % (what, entry.text, entry.value))
        if not apply_mod.rewritable(site)[0]:
            continue
        text = _canonical_addrs(site, apply_mod.addrs_ints(site))
        if site.addrs_text == text:
            continue
        edits.append(apply_mod.Edit(*site.addrs_span, text, "%s addrs" % site.site_id))
        resolved.append("%s addrs %s=%s" % (site.site_id, site.addrs_text, text))

    # Overlapping spans mean the source is not the shape this reader assumed, so there is no
    # canonical form to compare: `apply_edits` raises ApplyConflict, a ValueError, which
    # `same_program` already treats as undecidable.
    return ast.dump(ast.parse(apply_mod.apply_edits(src, edits))), resolved


def same_program(candidate_src, incumbent_src, defaults=None):
    """Return (equivalent, reason). An undecidable comparison lets the candidate proceed."""
    try:
        a, resolved = canonical(candidate_src)
        b, _ = canonical(incumbent_src)
    except (SyntaxError, ValueError, RecursionError) as exc:
        logger.debug("same_program could not decide: %s", exc)
        return False, ""
    if a != b:
        return False, ""
    if resolved:
        return True, ("restates a number the tuner owns: " + ", ".join(sorted(set(resolved))[:6]))
    return True, "identical once comments and formatting are removed"
