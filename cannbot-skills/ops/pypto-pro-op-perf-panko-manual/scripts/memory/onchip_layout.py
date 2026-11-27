# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Plan per-space tile placements subject to alignment, alias, depth and capacity constraints."""

from __future__ import annotations

import logging
from math import prod
import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default DMA segment alignment in bytes.
DEFAULT_ALIGNMENT = 32


class LayoutInvalid(ValueError):
    """The regions themselves do not describe a layout (bad alias, unknown width)."""


# Distinguish a complete memory-space ceiling from the allocation reserved
# for an individual author-owned region; their remedies are different.
REJECT_RULES = {
    "space_ceiling": "space ceiling: the chip's software-usable capacity for this space",
    "author_region": ("author region: the bytes this call's own `addrs=` reserves; this space is author-owned "
                      "and is not re-placed"),
}


class LayoutRejected(RuntimeError):
    """The placement exceeds a known capacity. The diagnostic identifies the space and footprint."""

    def __init__(self, space: str, needed: int, budget: int, rule: str = None):
        text = f"{space} needs {needed} bytes, budget is {budget} (over by {needed - budget})"
        if rule:
            text += f" [{REJECT_RULES.get(rule, rule)}]"
        super().__init__(text)
        self.space = space
        self.needed = needed
        self.budget = budget
        self.rule = rule


class LayoutOverlap(RuntimeError):
    """Raised when two regions overlap without a shared allocation."""


@dataclass(frozen=True)
class Region:
    """One `make_tile_group` worth of on-chip space."""

    name: str
    shape: Sequence[int]
    dtype: object
    space: str
    depth: int = 1
    alias_of: Optional[str] = None
    # Preserve the original base when overlapping windows encode a stride absent from the shape.
    fixed_addr: Optional[int] = None
    # Only author-owned layouts allow pre-existing overlaps.
    unchecked: bool = False
    # Author-owned tiles may shrink but cannot exceed their original size.
    author_bytes: Optional[int] = None


@dataclass(frozen=True)
class Placement:
    """Where a region went."""

    name: str
    space: str
    addr: int
    tile_bytes: int
    depth: int
    alias_of: Optional[str] = None
    #: True when this sits where the author put it rather than where packing would.
    fixed: bool = False
    #: True when its space is the author's and is exempt from the overlap check.
    unchecked: bool = False
    #: Independent JIT launches have separate physical arenas.
    scope: str = ""

    @property
    def size(self) -> int:
        return self.tile_bytes * self.depth

    @property
    def end(self) -> int:
        return self.addr + self.size


@dataclass(frozen=True)
class Layout:
    """A placement per region, with what it leaned on and what it could not."""

    placements: Tuple[Placement, ...]
    used: Mapping[str, int]
    budget: Mapping[str, Optional[int]]
    ungated: Tuple[str, ...]
    #: Spaces over the software-usable ceiling and within the physical buffer.
    #: Only ever non-empty when the caller passed `addr_budget` -- see `plan`.
    over_soft: Tuple[str, ...] = ()
    alignment: int = DEFAULT_ALIGNMENT
    notes: Tuple[str, ...] = ()
    #: Spaces whose addresses are the author's, kept as written because their map
    #: is not re-derivable from tile shapes. Tiles there may shrink, not grow.
    author_owned: Tuple[str, ...] = ()

    def addr(self, name: str) -> int:
        """The value to write as `addrs=` for this region."""
        return self._find(name).addr

    def stride(self, name: str) -> int:
        """Return the byte spacing between explicit per-depth tile addresses for the planned region."""
        p = self._find(name)
        if p.alias_of is not None:
            return self._find(p.alias_of).tile_bytes
        return p.tile_bytes

    def headroom(self, space: str) -> Optional[int]:
        """Bytes left in `space`, or None where there is no known ceiling."""
        budget = self.budget.get(space)
        if budget is None:
            return None
        return budget - self.used.get(space, 0)

    def _find(self, name: str) -> Placement:
        for p in self.placements:
            if p.name == name:
                return p
        raise KeyError(name)


# Source-level bit widths follow DataType::GetBit() in core/dtype.h.
DTYPE_BITS = {
    # 4-bit
    "DT_HF4": 4, "DT_FP4": 4, "DT_FP4E2M1": 4, "DT_FP4E1M2": 4,
    "DT_UINT4": 4, "DT_INT4": 4,
    # 8-bit -- DT_BOOL included, which `GetBit` reports as 8, not 1
    "DT_BOOL": 8, "DT_HF8": 8, "DT_FP8E4M3FN": 8, "DT_FP8E5M2": 8,
    "DT_FP8E8M0": 8, "DT_UINT8": 8, "DT_INT8": 8,
    # 16-bit
    "DT_BF16": 16, "DT_FP16": 16, "DT_UINT16": 16, "DT_INT16": 16,
    # 32-bit
    "DT_FP32": 32, "DT_FLOAT": 32, "DT_UINT32": 32, "DT_INT32": 32,
    # 64-bit
    "DT_UINT64": 64, "DT_INT64": 64, "DT_FP64": 64, "DT_INDEX": 64,
}

# Whole bytes for byte-width callers. Tile storage uses bits to preserve packed 4-bit types.
DTYPE_BYTES = {name: (bits + 7) // 8 for name, bits in DTYPE_BITS.items()}


#: Match complete dtype tokens: an unknown name can start with the name of a smaller valid dtype.
_DTYPE_TOKEN = re.compile(r"DT_[A-Za-z0-9_]+")

# DDR does not contribute to on-chip capacity.
_OFF_CHIP_SPACES = ("DDR",)


def dtype_tokens(dtype_text: object) -> Tuple[str, ...]:
    """Return distinct DT_ tokens in sorted order."""
    return tuple(sorted(set(_DTYPE_TOKEN.findall(str(dtype_text or "")))))


def bits_of_text(dtype_text: object) -> int:
    """Return dtype width in bits, or zero for an unknown dtype."""
    widths = {DTYPE_BITS[token] for token in dtype_tokens(dtype_text) if token in DTYPE_BITS}
    return widths.pop() if len(widths) == 1 else 0


def width_of_text(dtype_text: object) -> int:
    """Return the byte width of an exact dtype token, or zero for an unresolved type."""
    bits = bits_of_text(dtype_text)
    return (bits + 7) // 8 if bits > 0 else 0


def alias_key(site: object, addrs=None):
    """Identify an allocation by its memory space and complete sequence of resolved addresses."""
    space = getattr(site, "space", None)
    if addrs is None:
        addrs = getattr(site, "addrs_values", None) or ()
    if addrs and all(a is not None for a in addrs):
        return (space, tuple(int(a) for a in addrs))
    return (space, "text:" + ((getattr(site, "addrs_text", "") or "").strip()))


def group_bits(site: object, shape, depth: int = 1) -> int:
    """Count bits across all depth slots so packed types compare at their actual size."""
    n = prod(map(int, shape))
    return n * (bits_of_text(dtype_of(site)) or 8) * max(1, int(depth or 1))


def alias_hosts(groups):
    """Choose the largest alias-group member as host so that every member fits within its reserved span."""
    # max() returns the first of equal sizes, so ties keep insertion order.
    return {key: max((m for m in members if m[0] > -1), key=lambda m: m[0], default=(-1, None))[1]
            for key, members in groups.items()}


def dtype_of(site: object) -> str:
    """Resolve one unambiguous dtype token, including aliases."""
    text = getattr(site, "dtype_text", "") or ""
    if len(dtype_tokens(text)) == 1:
        return text
    return getattr(site, "dtype_resolved", "") or text


def _bytes_of(dtype: object) -> int:
    """Width via pypto, never a table here. 0 when it cannot be established."""
    try:
        from pypto import pypto_impl

        return int(pypto_impl.BytesOf(dtype))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("BytesOf(%r) unavailable: %s", dtype, exc)
        return 0


def tile_bytes(region: Region, width: Callable[[object], int] = _bytes_of) -> int:
    """Compute the Pro slot size including packed elements and required alignment."""
    if region.depth < 1:
        raise LayoutInvalid(f"region {region.name!r}: depth must be >= 1, got {region.depth}")
    if not region.shape or any(d < 1 for d in region.shape):
        raise LayoutInvalid(f"region {region.name!r}: bad shape {tuple(region.shape)!r}")
    n = prod(map(int, region.shape))
    bits = bits_of_text(region.dtype)
    if bits > 0:
        return (n * bits + 7) // 8
    w = width(region.dtype)
    if w <= 0:
        raise LayoutInvalid(f"region {region.name!r}: no byte width for dtype {region.dtype!r}; "
                            "refusing to size it as zero")
    return n * w


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return -(-value // alignment) * alignment


def _check_disjoint(placements: Iterable[Placement]) -> None:
    """Check that independent regions do not overlap. Permit sharing only within declared alias groups."""
    by_space: Dict[str, list] = {}
    for p in placements:
        by_space.setdefault((p.scope, p.space), []).append(p)

    for (_scope, space), group in by_space.items():
        if group and all(p.unchecked for p in group):
            continue
        hosts = {p.name: p for p in group if p.alias_of is None}
        standalone = sorted(hosts.values(), key=lambda p: p.addr)
        for earlier, later in zip(standalone, standalone[1:]):
            if later.addr < earlier.end:
                raise LayoutOverlap(f"{space}: {earlier.name} [{earlier.addr}, {earlier.end}) overlaps "
                                    f"{later.name} [{later.addr}, {later.end}); neither declared the other")
        for p in group:
            if p.alias_of is None:
                continue
            host = hosts.get(p.alias_of)
            if host is None:
                raise LayoutInvalid(f"{p.name!r} aliases {p.alias_of!r}, which is not a standalone region in {space}")
            if p.addr != host.addr:
                raise LayoutOverlap(f"{space}: {p.name} declares an alias of {host.name} but sits at "
                                    f"{p.addr}, not {host.addr}")
            if p.size > host.size:
                raise LayoutInvalid(f"{p.name} ({p.size} bytes) does not fit inside its alias host "
                                    f"{host.name} ({host.size} bytes)")


def allocation_groups(sites):
    """Partition independent launches. Keep sites with shared or unknown ownership together."""
    sites = tuple(sites)
    owners = [set(getattr(s, "allocation_scope", ()) or ()) for s in sites]
    if not sites or any(not scope for scope in owners):
        return [sites] if sites else []
    components = []
    for scope in owners:
        connected = [c for c in components if c & scope]
        for c in connected:
            scope = scope | c
            components.remove(c)
        components.append(scope)
    groups = [[] for _ in components]
    for site, scope in zip(sites, owners):
        index = next(i for i, component in enumerate(components) if component & scope)
        groups[index].append(site)
    return [tuple(group) for group in groups]


def combine_layouts(layouts):
    """Independent launch arenas share capacity, so their high-water is a max."""
    layouts = list(layouts)
    if len(layouts) == 1:
        return layouts[0]
    used, budget, placements = {}, {}, []
    for index, layout in enumerate(layouts):
        budget.update(layout.budget)
        for memory, size in layout.used.items():
            used[memory] = max(used.get(memory, 0), size)
        placements.extend(replace(p, scope=str(index)) for p in layout.placements)
    return Layout(placements=tuple(placements), used=used, budget=budget,
                  ungated=tuple(sorted({s for l in layouts for s in l.ungated})),
                  over_soft=tuple(sorted({s for l in layouts for s in l.over_soft})),
                  alignment=layouts[0].alignment if layouts else DEFAULT_ALIGNMENT,
                  notes=tuple(note for l in layouts for note in l.notes),
                  author_owned=tuple(sorted({s for l in layouts for s in l.author_owned})))


def author_tile_bytes(site: object):
    """One tile of `site` as the AUTHOR sized it, or None when unreadable."""
    shape = [getattr(e, "value", e) for e in getattr(site, "shape", ()) or ()]
    if not shape or any(v is None for v in shape):
        return None
    bits = bits_of_text(dtype_of(site))
    return (prod(map(int, shape)) * bits + 7) // 8 if bits > 0 else None


def _written_addrs(site: object, addrs_of=None):
    """Return numeric addresses from addrs_of or the site, or None when unresolved."""
    if addrs_of is not None:
        written = addrs_of(site)
        return list(written) if written else None
    values = getattr(site, "addrs_values", None) or ()
    if values and all(v is not None for v in values):
        return [int(v) for v in values]
    return None


def declared_regions(sites, *, addrs_of=None):
    """Read existing on-chip allocations at their declared addresses, including shared windows."""
    regions, unresolved = [], []
    for site in sites:
        if site.space in _OFF_CHIP_SPACES:
            continue
        dimensions = _region_dimensions(site)
        addresses = _written_addrs(site, addrs_of)
        if not dimensions or not addresses or any(a is None for a in addresses):
            unresolved.append(site.site_id)
            continue
        shape, dtype, depth = dimensions
        if any(address < 0 for address in addresses):
            raise LayoutInvalid(f"{site.site_id}: negative tile address")
        if not site.addrs_is_sequence:
            # A scalar base declares consecutive buffers, each one tile wide.
            regions.append(Region(site.site_id, shape, dtype, site.space, depth,
                                  fixed_addr=addresses[0], unchecked=True))
            continue
        if len(addresses) != depth:
            raise LayoutInvalid(f"{site.site_id}: original address count differs from depth")
        for index, address in enumerate(addresses):
            regions.append(Region(f"{site.site_id}[{index}]", shape, dtype, site.space,
                                  fixed_addr=address, unchecked=True))
    return regions, unresolved


def author_spans(sites, addrs_of=None):
    """Return declared byte spans per space, including explicit addresses and every depth slot."""
    spans = {}
    for site in sites:
        space = getattr(site, "space", None)
        if space is None or space in _OFF_CHIP_SPACES:
            continue
        tile = author_tile_bytes(site)
        if tile is None:
            continue
        written = _written_addrs(site, addrs_of)
        if not written:
            continue
        depth = getattr(site, "depth", None)
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            continue
        # A scalar base implies consecutive slots at one tile stride.
        bases = written if len(written) > 1 else [written[0] + i * tile for i in range(depth)]
        spans.setdefault(space, []).append((min(bases), max(bases) + tile, getattr(site, "site_id", "?")))
    return spans


def author_owned(sites, addrs_of=None):
    """Return spaces whose authored address map cannot be reconstructed and must be preserved."""
    return spaces_not_re_derivable(author_spans(sites, addrs_of))


def _ordered_alias_regions(fixed, packed, groups):
    """Emit each alias group's largest member before the first view that needs it."""
    hosts = alias_hosts(groups)
    regions = {region.name: region for region, _ in packed}
    result, placed = list(fixed), set()
    for region, key in packed:
        host = hosts[key]
        if host not in placed:
            result.append(regions[host])
            placed.add(host)
        if region.name not in placed:
            result.append(replace(region, alias_of=host))
            placed.add(region.name)
    return result


def _region_dimensions(site):
    """Read a site's sized shape, dtype and depth, or leave it unresolved."""
    shape = [e.value for e in site.shape]
    dtype = dtype_of(site)
    width = width_of_text(dtype)
    if site.space is None or not shape or any((d is None for d in shape)):
        return None
    if width <= 0:
        return None
    depth = site.depth
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        return None
    return shape, dtype, depth


def regions_from_sites(sites, author_sites=None, *, dimensions=_region_dimensions, addrs_of=None, pinned_base=min):
    """Build placement regions from declared or candidate dimensions, retaining authored aliases."""
    sites = tuple(sites)
    author = sites if author_sites is None else tuple(author_sites)
    owned = author_owned(author, addrs_of)
    original_sizes = {site.site_id: author_tile_bytes(site) for site in author}
    for space, reason in owned.items():
        logger.info("%s keeps the author's addresses: %s", space, reason)

    fixed, packed, unsized = [], [], []
    groups = {}
    for site in sites:
        dims = dimensions(site)
        if dims is None:
            unsized.append(site.site_id)
            continue
        shape, dtype, depth = dims
        region = Region(site.site_id, shape, dtype, site.space, depth)
        written = _written_addrs(site, addrs_of)
        if site.space in owned:
            if not written:
                unsized.append(site.site_id)
                continue
            size = original_sizes.get(site.site_id)
            if addrs_of is not None and site.site_id not in original_sizes:
                size = author_tile_bytes(site)
            fixed.append(replace(region, fixed_addr=pinned_base(written), author_bytes=size, unchecked=True))
        else:
            key = alias_key(site, written)
            groups.setdefault(key, []).append((group_bits(site, shape, depth), site.site_id))
            packed.append((region, key))

    return _ordered_alias_regions(fixed, packed, groups), unsized


def spaces_not_re_derivable(spans: Mapping[str, Sequence[Tuple[int, int, str]]]):
    """Identify memory spaces whose original address map cannot be repacked, with the reason for each."""
    out: Dict[str, str] = {}
    for space, ranges in (spans or {}).items():
        conflict = _different_start_overlap(sorted(ranges))
        if conflict is not None:
            a, b = conflict
            out[space] = (f"{a[2]} [{a[0]}, {a[1]}) and {b[2]} [{b[0]}, {b[1]})"
                          " overlap from different starts, so the map is not a function of the tile shapes")
    return out


def _different_start_overlap(ranges):
    for index, first in enumerate(ranges):
        for second in ranges[index + 1:]:
            if second[0] >= first[1]:
                break
            if first[0] != second[0]:
                return first, second
    return None


def _place_region(region, placements, high_water, alignment, width):
    """Resolve a fixed, aliased or packed placement and update its arena."""
    size = tile_bytes(region, width) * region.depth
    slot = {"name": region.name, "space": region.space, "tile_bytes": size // region.depth, "depth": region.depth}
    if region.fixed_addr is not None:
        author = region.author_bytes
        if region.unchecked and author is not None and size > author * region.depth:
            raise LayoutRejected(region.space, size, author * region.depth, rule="author_region")
        end = region.fixed_addr + size
        if end > high_water.get(region.space, 0):
            high_water[region.space] = end
        return Placement(addr=region.fixed_addr, fixed=True, unchecked=region.unchecked, **slot)
    if region.alias_of is not None:
        host = next((p for p in placements if p.name == region.alias_of), None)
        if host is None:
            raise LayoutInvalid(f"{region.name!r} aliases {region.alias_of!r}, which is not placed before it")
        if host.space != region.space:
            raise LayoutInvalid(f"{region.name!r} aliases {region.alias_of!r} across memory spaces "
                                f"({region.space} vs {host.space})")
        return Placement(addr=host.addr, alias_of=region.alias_of, **slot)
    base = _align_up(high_water.get(region.space, 0), alignment)
    high_water[region.space] = base + size
    return Placement(addr=base, **slot)


def _check_capacity(high_water, budget, addr_budget, notes):
    """Enforce software ceilings and report explicit physical-bound allowances."""
    soft = []
    ungated: list = []
    for space, used in high_water.items():
        ceiling = budget.get(space)
        physical = (addr_budget or {}).get(space)
        if physical is not None and used > physical:
            raise LayoutRejected(space, used, physical, rule="physical address limit")
        if ceiling is None:
            ungated.append(space)
            notes.append(f"{space} placed with no capacity to check against; the disjointness "
                         "check is the only bound on this layout")
            continue
        if used > ceiling:
            if physical is not None:
                # Report layouts above the software ceiling that still fit the physical buffer.
                soft.append(space)
                notes.append(f"{space} uses {used} bytes, over the software-usable {ceiling} and within the "
                             f"physical address limit {physical}")
                continue
            raise LayoutRejected(space, used, ceiling, rule="space_ceiling")

    return ungated, soft


def plan(
    regions: Sequence[Region],
    budget: Mapping[str, Optional[int]],
    alignment: int = DEFAULT_ALIGNMENT,
    width: Callable[[object], int] = _bytes_of,
    addr_budget: Optional[Mapping[str, Optional[int]]] = None,
) -> Layout:
    """Place regions under capacity, alias and alignment constraints or raise a layout error."""
    placements, high_water, notes, seen = [], {}, [], set()
    for region in regions:
        if region.name in seen:
            raise LayoutInvalid(f"duplicate region name {region.name!r}")
        seen.add(region.name)
    for region in regions:
        placements.append(_place_region(region, placements, high_water, alignment, width))
    _check_disjoint(placements)
    ungated, soft = _check_capacity(high_water, budget, addr_budget, notes)
    owned = sorted({p.space for p in placements if p.unchecked})
    notes.extend(f"{space} keeps the addresses the author wrote: its map overlaps in a way "
                 "tile shapes cannot re-derive, so this module does not pack it" for space in owned)
    return Layout(placements=tuple(placements), used=dict(high_water), budget={s: budget.get(s) for s in high_water},
                  ungated=tuple(sorted(ungated)), over_soft=tuple(sorted(soft)), alignment=alignment,
                  notes=tuple(notes), author_owned=tuple(owned))
