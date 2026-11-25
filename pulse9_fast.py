#!/usr/bin/env python3
"""
Ada40 / Pulse 9 + Rule 110 — speed & memory optimized
Adam Lee Hatchett – 25 Nov 2026

- Integer lattice only (uint8)
- Pre-allocated grids
- Vectorized Rule 110
- Single-pass pressure + mask
- No SciPy required for the hot path
- Optional lightweight neural prior
"""

import numpy as np
from pathlib import Path

# ---------- Fast Rule 110 ----------
_RULE110 = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)

def rule110_step(row):
    """Vectorized single step. row is uint8 1-D."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return _RULE110[idx]

def evolve_r110(init, steps):
    """Pre-allocated evolution. Returns (steps, N) uint8."""
    n = len(init)
    grid = np.empty((steps, n), dtype=np.uint8)
    grid[0] = init
    for t in range(1, steps):
        grid[t] = rule110_step(grid[t - 1])
    return grid

def make_ether(length):
    unit = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    return np.tile(unit, (length // 14) + 1)[:length]

# ---------- Fast pressure + mask (no SciPy) ----------
def fast_pressure_and_mask(grid, late=32):
    dens = grid.astype(np.float32)
    smooth = (np.roll(dens, 1, axis=1) + dens + np.roll(dens, -1, axis=1)) / 3.0
    pressure = np.abs(np.diff(smooth, axis=1, prepend=smooth[:, :1]))

    late_mean = dens[-late:].mean(axis=0)
    mask = (np.roll(late_mean, 1) + late_mean + np.roll(late_mean, -1)) / 3.0
    mask /= (mask.max() + 1e-8)
    return pressure, mask

def container_weights(mask, pressure_slice, n=4):
    pmax = pressure_slice.max() + 1e-8
    bins = np.minimum((pressure_slice / pmax * n).astype(np.int32), n - 1)
    weights = np.zeros(n, dtype=np.float32)
    for c in range(n):
        weights[c] = (mask * (bins == c)).sum()
    weights /= (weights.sum() + 1e-8)
    return weights

def apply_prior(features, mask, strength=0.35):
    m = np.interp(
        np.linspace(0, 1, features.shape[1]),
        np.linspace(0, 1, len(mask)),
        mask
    ).astype(np.float32)
    features *= (1.0 + strength * (m - 1.0))
    return features

def main():
    np.random.seed(7)
    N = 256
    STEPS = 128

    ether = make_ether(N)
    init = ether.copy()
    init[32:38] = 1
    init[96:104] = [1, 1, 1, 1, 1, 0, 1, 0]
    init[160] = 1 - init[160]

    grid = evolve_r110(init, STEPS)
    pressure, mask = fast_pressure_and_mask(grid)
    weights = container_weights(mask, pressure[-1])

    print("Container weights:", np.round(weights, 3))
    print("Mask mean:", round(float(mask.mean()), 3))
    print("Grid shape / dtype:", grid.shape, grid.dtype)
    print("Peak memory estimate: ~", grid.nbytes / 1024, "KB for lattice")

    batch, dim = 32, 64
    feats = np.random.randn(batch, dim).astype(np.float32)
    apply_prior(feats, mask)
    print("Features modulated. Mean abs:", round(float(np.abs(feats).mean()), 4))

if __name__ == '__main__':
    main()
