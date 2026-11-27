# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Require an inspectable baseline before starting shape optimization."""
from bayesian_optimization import apply, block, policy, space
from memory import onchip_layout


def original_layout(sites, hw):
    """Check original addresses without repacking. Keep separate address slots and author-provided overlaps."""
    return onchip_layout.combine_layouts(
        onchip_layout.plan(onchip_layout.declared_regions(group, addrs_of=apply.addrs_ints)[0], hw.budget(),
                           alignment=hw.alignment, width=hw.width_fn())
        for group in onchip_layout.allocation_groups(sites))


def check(op_file, source, envelope=None, constants=None, live_tiles=None):
    report = {"ok": False, "reasons": [], "source_unchanged": True}
    try:
        hw, domain = block.hw_for(op_file, live_tiles, envelope, author_src=source, extra_consts=constants)
        program = apply.tile_program(source, hw.extra_consts)
        sites = program.active_sites
        report["declaration_count"] = len(sites)
        refusals = apply.refusals(program)
        refusals.update((site.site_id, "element dtype width is unresolved") for site in sites
                        if onchip_layout.width_of_text(onchip_layout.dtype_of(site)) <= 0)
        report["unresolved_declarations"] = refusals
        if not sites:
            report["reasons"].append(
                "No tile declarations are visible for shape TPE; provide the kernel/case context "
                "or explicitly disable tile search.")
        if refusals:
            report["reasons"].append("Normalize or bind all live tile declarations before starting shape TPE.")
        if report["reasons"]:
            return report
        shape_sites = policy.shape_sites(apply.tunable_sites(program))
        anchor = policy.shape_config(apply.incumbent_config(program), shape_sites)
        domains = space.grouped_domains(policy.shared_shape_groups(shape_sites, domain=domain), shape_sites,
                                        anchor, domain, apply.tie_groups(shape_sites, anchor))
        report["shape_parameter_count"] = len(domains)
        report["variable_shape_parameter_count"] = sum(len(values) > 1 for values in domains.values())
        report["ungated_spaces"] = sorted({s.space for s in sites
                                           if s.space != "DDR" and hw.budget().get(s.space) is None})
        if not domains or any(not values for values in domains.values()):
            report["reasons"].append("The baseline has no complete shape search domain.")
        report["baseline_layout_valid"] = False
        original_layout(sites, hw)
        report["baseline_layout_valid"] = True
        report["baseline_layout_method"] = "Exact declared buffer addresses; no packing; original overlaps retained."
    except (ValueError, TypeError, SyntaxError, onchip_layout.LayoutRejected, onchip_layout.LayoutOverlap) as exc:
        report["reasons"].append(str(exc))
    report["ok"] = not report["reasons"]
    return report
