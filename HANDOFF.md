# Comprehensive Handoff
**Ada40 / Pulse 9 Lattice Pressure Engine**
Adam Lee Hatchett
25 November 2026

## 1. Core Claim
Local lattice systems (Rule 110, hexagonal FHP lattice gas, or D2Q9 Lattice Boltzmann) generate a density field. That density field is treated as a pressure field. Its late-time average becomes a soft mask; its spatial gradient drives adaptive container switching. The same pressure signal is used as a neural prior. This combination is the documented contribution.

## 2. Lattice Engines (Interchangeable)
| Engine | State | Continuum Limit | Memory | Best Use |
|--------|-------|------------------|--------|----------|
| Rule 110 | Boolean | Universal computation + gliders | Lowest | Edge / ESP32 |
| Hexagonal FHP | Boolean (6 dirs) | Navier–Stokes (noisy) | Low | Fluid analogy on Boolean hardware |
| D2Q9 LBM (BGK/TRT/MRT) | float32 distributions | Clean Navier–Stokes | Higher | Quantitative pressure, GPU |

All three produce a density array that is used identically downstream.

## 3. Pressure Extraction
```python
density = sum(occupancy or f_i)
pressure_diff = abs(neighbor differences)
soft_mask = late_row_average(density)
```

## 4. Pulse 9 Layer
- Soft mask = normalized late-time density average
- Container weights = softmax over binned pressure differentials
- Adaptive switching driven by local ∇p
- Neural prior = element-wise modulation by the soft mask

## 5. Collision Operators
- BGK: simplest, single τ
- TRT: two rates, magic parameter Λ=1/4 recommended
- MRT: full moment-space control, highest stability

All output the same density interface.

## 6. GPU Notes
Memory-bandwidth bound. Shared-memory tiling + bank-conflict padding approaches 85–95% of STREAM.

## 7. Attribution
Adam Lee Hatchett / Ada40, November 2026.
