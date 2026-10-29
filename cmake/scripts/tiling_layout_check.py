#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""tiling 结构字节级对齐 构建期自动校验.

TilingData 存在两份互不相干的声明——
  1. pypto codegen 生成的 <X>_tiling.h: 按 kernel dataclass 声明顺序生成, 是权威布局;
  2. op_host 手写 BEGIN_TILING_DATA_DEF + REGISTER_TILING_DATA_CLASS: 框架按它分配 tiling buffer.
两份漂移不会有任何编译/运行报错, 只会让 kernel 读到整体错位的字段.

本脚本 (configure 期由 add_tiling_modules 调用, 早于一切编译):
  a) 解析两侧字段序列 (名称/类型/数组长度) 做 diff, 不一致即打印 diff 并退出码 1 (报错即拒);
  b) 一致则生成 tiling 布局核对程序 (custom_build.cmake 构建并在 tiling 编译前执行):
     运行时构造手写 TilingData, 以 TilingDef::GetDataSize()/GetFieldInfo() 读取框架真实
     buffer 布局, 与编译期 sizeof(kernel dataclass 生成类) 及期望字段序列交叉验证 ——
     padding 级漂移 (如 float 字段后的对齐填充) 由框架布局机制兜底钉死.

注: 手写侧布局由 BEGIN_TILING_DATA_DEF 的 FieldHandler 在运行时累加 (C++ 对象尺寸含
    每字段 offset+16B reserve 三件套, sizeof 不可比), 故编译期 offsetof/static_assert
    对手写侧不可行; gym 采用运行时核对程序补全该侧校验.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Field:
    ctype: str
    name: str
    array_len: int  # 0 = 标量


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _norm_type(t: str) -> str:
    return re.sub(r"\s+", "", t)


def parse_generated_header(gen_dir: Path) -> tuple:
    """解析 codegen 生成的 <X>_tiling.h (单一 @pl.jit kernel 契约: 恰好一个)."""
    candidates = sorted(gen_dir.glob("*_tiling.h"))
    if not candidates:
        raise SystemExit(f"[tiling_check] no *_tiling.h under {gen_dir}")
    if len(candidates) > 1:
        raise SystemExit(
            f"[tiling_check] multiple *_tiling.h under {gen_dir}: "
            f"{[p.name for p in candidates]} (one @pl.jit kernel per entry file)")
    text = candidates[0].read_text()
    m = re.search(r"(?:class|struct)\s+(\w+)\s*\{[^}]*?public:\s*(.*?)\s*\};", text, re.S)
    if not m:
        raise SystemExit(f"[tiling_check] cannot parse generated header {candidates[0]}")
    gen_name, body = m.group(1), m.group(2)
    fields = []
    for tok in body.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        fm = re.match(r"^([\w:]+)\s+(\w+)\s*(?:\[(\d+)\])?$", tok)
        if not fm:
            raise SystemExit(f"[tiling_check] unparsed field '{tok}' in {candidates[0]}")
        fields.append(Field(_norm_type(fm.group(1)), fm.group(2), int(fm.group(3) or 0)))
    if not fields:
        raise SystemExit(f"[tiling_check] empty field list in {candidates[0]}")
    return gen_name, fields, candidates[0]


def parse_handwritten(host_dir: Path) -> tuple:
    """解析 op_host 手写 TilingData: REGISTER_TILING_DATA_CLASS + BEGIN_TILING_DATA_DEF."""
    srcs = sorted(p for p in host_dir.rglob("*") if p.suffix in (".h", ".cpp"))
    reg_re = re.compile(r"REGISTER_TILING_DATA_CLASS\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
    regs = []
    for p in srcs:
        for m in reg_re.finditer(_strip_comments(p.read_text())):
            regs.append((p, m.group(1), m.group(2)))
    if not regs:
        raise SystemExit(
            f"[tiling_check] no REGISTER_TILING_DATA_CLASS in {host_dir}; "
            "pypto built-in op must register its tiling struct")
    structs = {(op, s) for _, op, s in regs}
    if len(structs) > 1:
        raise SystemExit(
            f"[tiling_check] ambiguous registrations {sorted(structs)} in {host_dir} "
            "(gym contract: one op per op_host dir)")
    reg_file, op_type, struct = regs[0]

    begin_re = re.compile(
        rf"BEGIN_TILING_DATA_DEF\s*\(\s*{struct}\s*\)(.*?)END_TILING_DATA_DEF", re.S)
    block = None
    block_file = None
    for p in srcs:
        m = begin_re.search(_strip_comments(p.read_text()))
        if m:
            block, block_file = m.group(1), p
            break
    if block is None:
        raise SystemExit(f"[tiling_check] BEGIN_TILING_DATA_DEF({struct}) not found in {host_dir}")

    field_re = re.compile(
        r"TILING_DATA_FIELD_DEF(?:_ARR)?\s*\([^()]*\)")
    scalar_re = re.compile(r"TILING_DATA_FIELD_DEF\s*\(\s*([^,]+?)\s*,\s*(\w+)\s*\)")
    arr_re = re.compile(
        r"TILING_DATA_FIELD_DEF_ARR\s*\(\s*([^,]+?)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)")
    fields = []
    residue = field_re.sub("", block)
    if re.search(r"\S", residue):
        unknown = re.search(r"\S[^\n]*", residue).group(0).strip()
        raise SystemExit(
            f"[tiling_check] unsupported construct '{unknown}' in {block_file} "
            "(extend tiling_layout_check.py for nested/struct fields)")
    for m in field_re.finditer(block):
        stmt = m.group(0)
        am = arr_re.fullmatch(stmt)
        if am:
            fields.append(Field(_norm_type(am.group(1)), am.group(3), int(am.group(2))))
            continue
        sm = scalar_re.fullmatch(stmt)
        if sm:
            fields.append(Field(_norm_type(sm.group(1)), sm.group(2), 0))
        else:
            raise SystemExit(f"[tiling_check] unparsed TILING_DATA stmt '{stmt}' in {block_file}")

    ns = "optiling" if re.search(r"namespace\s+optiling", block_file.read_text()) else ""
    return op_type, struct, fields, block_file, ns


def diff_fields(op_type, gen_name, gen_fields, hw_struct, hw_fields):
    issues = []
    if len(gen_fields) != len(hw_fields):
        issues.append(
            f"  field count: kernel dataclass {len(gen_fields)} vs handwritten {len(hw_fields)}")
    for i in range(max(len(gen_fields), len(hw_fields))):
        g = gen_fields[i] if i < len(gen_fields) else None
        h = hw_fields[i] if i < len(hw_fields) else None
        if g is None or h is None:
            issues.append(f"  [{i}] kernel={g or '<missing>'} vs handwritten={h or '<missing>'}")
            continue
        if g.name != h.name:
            issues.append(f"  [{i}] name: kernel '{g.name}' vs handwritten '{h.name}'")
        if g.ctype != h.ctype:
            issues.append(f"  [{i}] {g.name}: type kernel '{g.ctype}' vs handwritten '{h.ctype}'")
        if g.array_len != h.array_len:
            issues.append(
                f"  [{i}] {g.name}: array len kernel [{g.array_len}] vs handwritten [{h.array_len}]")
    if issues:
        print(f"[tiling_check] tiling layout check FAILED for {op_type}: "
              f"handwritten {hw_struct} != kernel dataclass {gen_name}", file=sys.stderr)
        print("  kernel side   : " + ", ".join(
            f"{f.ctype} {f.name}" + (f"[{f.array_len}]" if f.array_len else "") for f in gen_fields),
            file=sys.stderr)
        print("  handwritten   : " + ", ".join(
            f"{f.ctype} {f.name}" + (f"[{f.array_len}]" if f.array_len else "") for f in hw_fields),
            file=sys.stderr)
        for issue in issues:
            print(issue, file=sys.stderr)
        print("  fix: sync op_host tiling struct with op_kernel dataclass "
              "(field NAMES, ORDER and TYPES must match; PyPTO int -> int64_t, float -> float)",
              file=sys.stderr)
        return False
    return True


def emit_check_program(path: Path, hw_header: Path, gen_header: Path, ns, hw_struct,
                       gen_name, fields, op_type):
    qual = f"{ns}::{hw_struct}" if ns else hw_struct
    names = ", ".join(f'"{f.name}"' for f in fields)
    dtypes = ", ".join(f'"{f.ctype}"' for f in fields)
    arrs = ", ".join(str(f.array_len) for f in fields)
    lines = [
        f"// Auto-generated by tiling_layout_check.py (tiling layout contract, op {op_type}). Do not edit.",
        "#include <cstddef>",
        "#include <cstdio>",
        "#include <cstring>",
        f'#include "{hw_header}"',
        f'#include "{gen_header}"',
        "",
        "namespace {",
        "bool IsPaddingPlaceholder(const char *name) {",
        "    const size_t len = std::strlen(name);",
        "    return len > 2 && std::strcmp(name + len - 2, \"PH\") == 0;",
        "}",
        "void DumpRegistered(const std::vector<optiling::FieldInfo> &registered) {",
        "    for (const auto &f : registered) {",
        "        std::fprintf(stderr, \"    registered: name=%s dtype=%s classType=%s%s\\n\",",
        "                     f.name_, f.dtype_, f.classType_,",
        "                     std::strcmp(f.classType_, \"1\") == 0 ? \" (array)\" : \"\");",
        "    }",
        "}",
        "}",
        "",
        "int main() {",
        f"    {qual} td;",
        f"    const size_t kernelSize = sizeof(::{gen_name});",
        "    if (td.GetDataSize() != kernelSize) {",
        "        std::fprintf(stderr,",
        f"            \"[tiling_check] tiling layout check FAILED {op_type}: tiling buffer size mismatch: \"",
        f"            \"host {hw_struct}::GetDataSize()=%zu vs kernel sizeof({gen_name})=%zu; \"",
        "            \"sync op_host tiling struct with op_kernel dataclass\\n\",",
        "            td.GetDataSize(), kernelSize);",
        "        return 1;",
        "    }",
        "    const auto registered = td.GetFieldInfo();",
        f"    static const char *kNames[] = {{{names}}};",
        f"    static const char *kDtypes[] = {{{dtypes}}};",
        f"    static const size_t kArrLens[] = {{{arrs}}};",
        f"    const size_t kCount = {len(fields)};",
        "    // 框架为对齐 padding 插入 \"xxxPH\" uint8_t 占位字段, 序列比对仅看真实字段",
        "    size_t realCount = 0;",
        "    for (const auto &f : registered) {",
        "        if (!IsPaddingPlaceholder(f.name_)) { ++realCount; }",
        "    }",
        "    if (realCount != kCount) {",
        "        std::fprintf(stderr,",
        f"            \"[tiling_check] tiling layout check FAILED {op_type}: registered real field count %zu != expected %zu\\n\",",
        "            realCount, kCount);",
        "        DumpRegistered(registered);",
        "        return 1;",
        "    }",
        "    size_t idx = 0;",
        "    for (const auto &f : registered) {",
        "        if (IsPaddingPlaceholder(f.name_)) { continue; }",
        "        const bool isArray = kArrLens[idx] != 0;",
        "        const bool fieldIsScalar = std::strcmp(f.classType_, \"0\") == 0;",
        "        const bool fieldIsArray = std::strcmp(f.classType_, \"1\") == 0;",
        "        if (std::strcmp(f.name_, kNames[idx]) != 0 ||",
        "            std::strcmp(f.dtype_, kDtypes[idx]) != 0 ||",
        "            isArray ? !fieldIsArray : !fieldIsScalar ||",
        "            (isArray && f.arrSize_ != kArrLens[idx])) {",
        "            std::fprintf(stderr,",
        f"                \"[tiling_check] tiling layout check FAILED {op_type}: field[%zu] registered as \"",
        "                \"(name=%s dtype=%s classType=%s) but expected (name=%s dtype=%s arr=%zu)\\n\",",
        "                idx, f.name_, f.dtype_, f.classType_,",
        "                kNames[idx], kDtypes[idx], kArrLens[idx]);",
        "            DumpRegistered(registered);",
        "            return 1;",
        "        }",
        "        ++idx;",
        "    }",
        "    std::printf(\"[tiling_check] OK %s: %zu fields, %zu-byte tiling buffer \"",
        f"                \"(host {hw_struct} <-> kernel {gen_name})\\n\", \"{op_type}\", kCount, kernelSize);",
        "    return 0;",
        "}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="tiling layout byte-alignment contract check")
    parser.add_argument("--gen-dir", required=True, help="pypto codegen dir (PYPTO_GEN_DIR)")
    parser.add_argument("--host-dir", required=True, help="operator op_host dir")
    parser.add_argument("--op-file", required=True, help="op file stem, e.g. flex_attention")
    parser.add_argument("--emit-dir", required=True, help="dir for the static_assert TU")
    args = parser.parse_args()

    gen_name, gen_fields, gen_header = parse_generated_header(Path(args.gen_dir))
    op_type, hw_struct, hw_fields, hw_header, ns = parse_handwritten(Path(args.host_dir))

    if not diff_fields(op_type, gen_name, gen_fields, hw_struct, hw_fields):
        raise SystemExit(1)

    tu = Path(args.emit_dir) / f"{args.op_file}_tiling_layout_check.cpp"
    emit_check_program(tu, hw_header.resolve(), gen_header.resolve(), ns, hw_struct,
                       gen_name, gen_fields, op_type)
    print(f"[tiling_check] OK {op_type}: {len(gen_fields)} fields consistent "
          f"({gen_name} <-> {hw_struct}); check program -> {tu}")


if __name__ == "__main__":
    main()
