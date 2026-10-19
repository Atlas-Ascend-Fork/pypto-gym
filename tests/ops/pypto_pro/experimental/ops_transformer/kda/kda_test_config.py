import os
import sys
import logging

import torch
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _HERE
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = _REPO_ROOT
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ops.pypto_pro.experimental.ops_transformer.kda.kda_common import (
    C, K, V, HC, K_DIM, V_DIM, DEVICE,
    make_cu_seqlens_tensor, build_chunk_tables, alloc_chunk_h_workspaces,
)

CHUNK_SIZE = C
HV = 4

DEVICE_ID = os.environ.get('TILE_FWK_DEVICE_ID', '0')

TEST_SHAPES = [
    (8192, None),
    (4117, None),
    (768, [0, 128, 384, 768]),
    (156, [0,63,156]),
]


def require_a5(device=None):
    if device is None:
        device = DEVICE
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")


def make_kda_base_inputs(T, device):
    """Canonical KDA test inputs shared across all single-stage and e2e tests.

    Returns (q, k, v, g_log, beta_sig, scale) on ``device``.  Matches the
    chunk_kda (ling_3_0_flash) stability HARD CONSTRAINT composition
    (SPEC.md §9): q/k/v = randn*0.1, g = logsigmoid(randn) <= 0,
    beta = sigmoid(randn), seed 42, RNG order g0 -> q -> k -> v -> beta.

    Shapes:
        q:        [1, T, HV, K_DIM]  fp32  randn * 0.1
        k:        [1, T, HV, K_DIM]  fp32  randn * 0.1
        v:        [1, T, HV, V_DIM]  fp32  randn * 0.1
        g_log:    [1, T, HV, K_DIM]  fp32  logsigmoid(randn) <= 0
        beta_sig: [1, T, HV]         fp32  sigmoid(randn)
        scale:    float               K_DIM**-0.5
    """
    torch.manual_seed(42)
    g0 = torch.randn(1, T, HV, K_DIM, dtype=torch.float32)
    q = (torch.randn(1, T, HV, K_DIM, dtype=torch.float32) * 0.1).to(device)
    k = (torch.randn(1, T, HV, K_DIM, dtype=torch.float32) * 0.1).to(device)
    v = (torch.randn(1, T, HV, V_DIM, dtype=torch.float32) * 0.1).to(device)
    g_log = torch.nn.functional.logsigmoid(g0).clamp(min=-0.5).to(device)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV, dtype=torch.float32)).to(device)
    scale = K_DIM ** -0.5
    return q, k, v, g_log, beta_sig, scale


def make_tril_mask(C, device, diagonal=0):
    return torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=diagonal).to(device)


def run_main(title, shapes, run_case_fn):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(title)
    logging.info("=" * 60)
    idx = None
    device = DEVICE
    for arg in sys.argv[1:]:
        if arg.isdigit():
            idx = int(arg) - 1
        elif arg == "cpu":
            device = "cpu"
    selected = [shapes[idx]] if idx is not None else shapes
    logging.info("Running %d/%d case(s): %s", len(selected), len(shapes), selected)
    for T, cu in selected:
        cu_str = f"cu={cu}" if cu else "cu=None"
        run_case_fn(T, cu, f"T={T}, {cu_str}", device)
