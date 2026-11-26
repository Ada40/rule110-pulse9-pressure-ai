#!/usr/bin/env python3
"""
TRT (Two-Relaxation-Time) collision for D2Q9 LBM
Adam Lee Hatchett / Ada40 – 25 Nov 2026
"""

import numpy as np

C = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.int32)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

class TRT_LBM:
    def __init__(self, nx, ny, tau_plus=0.8, magic=0.25):
        self.nx, self.ny = nx, ny
        self.tau_plus = np.float32(tau_plus)
        self.tau_minus = 0.5 + magic / (tau_plus - 0.5)
        self.omega_plus = 1.0 / self.tau_plus
        self.omega_minus = 1.0 / self.tau_minus

        self.f = np.zeros((9, nx, ny), dtype=np.float32)
        self.rho = np.ones((nx, ny), dtype=np.float32)
        self.ux = np.zeros((nx, ny), dtype=np.float32)
        self.uy = np.zeros((nx, ny), dtype=np.float32)

    def init_perturbation(self, radius=12):
        cx, cy = self.nx // 2, self.ny // 2
        yy, xx = np.ogrid[:self.nx, :self.ny]
        r2 = (xx - cx)**2 + (yy - cy)**2
        self.rho = 1.0 + 0.25 * np.exp(-r2 / 50.0)
        self.rho = self.rho.astype(np.float32)
        for k in range(9):
            self.f[k] = W[k] * self.rho

    def equilibrium(self, rho, ux, uy):
        feq = np.zeros((9, self.nx, self.ny), dtype=np.float32)
        u2 = ux*ux + uy*uy
        for k in range(9):
            cu = C[k,0]*ux + C[k,1]*uy
            feq[k] = W[k] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*u2)
        return feq

    def collide_trt(self):
        self.rho = self.f.sum(0)
        self.ux = (self.f * C[:,0,None,None]).sum(0) / self.rho
        self.uy = (self.f * C[:,1,None,None]).sum(0) / self.rho

        feq = self.equilibrium(self.rho, self.ux, self.uy)

        for k in range(9):
            k_opp = OPP[k]
            f_even = 0.5 * (self.f[k] + self.f[k_opp])
            f_odd  = 0.5 * (self.f[k] - self.f[k_opp])
            feq_even = 0.5 * (feq[k] + feq[k_opp])
            feq_odd  = 0.5 * (feq[k] - feq[k_opp])

            self.f[k] = (self.f[k]
                         - self.omega_plus  * (f_even - feq_even)
                         - self.omega_minus * (f_odd  - feq_odd))

    def stream(self):
        for k in range(9):
            self.f[k] = np.roll(np.roll(self.f[k], C[k,0], 0), C[k,1], 1)

    def step(self):
        self.collide_trt()
        self.stream()

if __name__ == '__main__':
    np.random.seed(42)
    lbm = TRT_LBM(64, 48, tau_plus=0.8, magic=0.25)
    lbm.init_perturbation()

    for step in range(100):
        lbm.step()
        if step % 25 == 0:
            print(f"Step {step:3d}  mean ρ={lbm.rho.mean():.4f}  max ρ={lbm.rho.max():.4f}")

    print("TRT finished. Final mean density:", float(lbm.rho.mean()))
    print("tau+ =", lbm.tau_plus, "  tau- =", lbm.tau_minus)
