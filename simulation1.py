# r1 vs k1 code

import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from concurrent.futures import ProcessPoolExecutor
import numba as nb

class OscillatorsSimulator:
    def __init__(self, k1_start, k1_end, k2, n, tran, niter, h, dk):
        self.k1_start = k1_start
        self.k1_end = k1_end
        self.k2 = k2
        self.n = n
        self.tran = tran
        self.niter = niter
        self.h = h
        self.dk = dk

    def simulate(self):
        pi = np.arctan(1.0) * 4
        random_state = np.random.RandomState(1234568)
        omega = np.tan((np.arange(self.n) * pi) / self.n - ((self.n + 1) * pi) / (2 * self.n))

        theta = -1.0 * pi + 2.0 * pi * random_state.rand(self.n)
        forward_results = self.run_simulation(theta, omega, self.k1_start, self.k1_end, self.dk, self.n, self.tran, self.niter, self.h, self.k2)

        theta = 2 * pi * np.ones(self.n)
        backward_results = self.run_simulation(theta, omega, self.k1_end, self.k1_start, -self.dk, self.n, self.tran, self.niter, self.h, self.k2)

        return {
            'k1_values_forward': forward_results[0],
            'r1_values_forward': forward_results[1],
            'k1_values_backward': backward_results[0],
            'r1_values_backward': backward_results[1],
        }

    def run_simulation(self, theta, omega, k1_start, k1_end, dk, n, tran, niter, h, k2):
        k1_values = []
        r1_values = []

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for K1 in np.arange(k1_start, k1_end + np.sign(dk) * 0.01, dk):
                futures.append(executor.submit(run_simulation_step, theta.copy(), omega, K1, k2, n, tran, niter, h))

            for future in futures:
                K1, r1 = future.result()
                k1_values.append(K1)
                r1_values.append(r1)

        return k1_values, r1_values

@nb.njit
def run_simulation_step(theta, omega, K1, K2, n, tran, niter, h):
    pi = np.arctan(1.0) * 4
    r1 = 0.0
    r2 = 0.0
    beta = 0.0
    alpha = 0.0
    for it in range(1, niter + 1):
        rc1 = np.cos(theta).sum()
        rs1 = np.sin(theta).sum()
        rc2 = np.cos(2 * theta).sum()
        rs2 = np.sin(2 * theta).sum()
        ra = np.sqrt(rs1 ** 2 + rc1 ** 2) / n
        rb = np.sqrt(rs2 ** 2 + rc2 ** 2) / n
        dth = np.zeros_like(theta)
        derivs(0, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, beta, alpha, n)
        tho = np.zeros_like(theta)
        rk4(theta, dth, n, 0, h, tho, omega, K1, K2, ra, rs1, rs2, rc1, rc2, beta, alpha)
        theta = np.mod(tho, 2 * pi)
        if it > tran:
            r1 += ra
            r2 += rb
    r1 /= niter - tran
    return K1, r1

@nb.njit
def derivs(t, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, beta, alpha, n):
    x = 0
    y = 0
    for i in range(n):
        dth[i] = omega[i] + (ra ** x) * (K1 / n) * (np.cos(theta[i] + alpha) * rs1 - np.sin(theta[i] + alpha) * rc1) + \
                 (K2 / n ** 2) * (ra ** y) * (np.cos(theta[i] + beta) * rs2 * rc1 - np.sin(theta[i] + beta) * rs2 * rs1 -
                                              np.sin(theta[i] + beta) * rc2 * rc1 - np.cos(theta[i] + beta) * rc2 * rs1)

@nb.njit
def rk4(y, dydx, n, x, h, yout, omega, K1, K2, ra, rs1, rs2, rc1, rc2, beta, alpha):
    dym = np.zeros_like(y)
    dyt = np.zeros_like(y)
    yt = np.zeros_like(y)
    hh = h * 0.5
    h6 = h / 6.0
    xh = x + hh
    yt = y + hh * dydx
    derivs(xh, yt, dyt, omega, K1, K2, rs1, rs2, rc1, rc2, ra, beta, alpha, n)
    yt = y + hh * dyt
    derivs(xh, yt, dym, omega, K1, K2, rs1, rs2, rc1, rc2, ra, beta, alpha, n)
    yt = y + h * dym
    dym += dyt
    derivs(x + h, yt, dyt, omega, K1, K2, rs1, rs2, rc1, rc2, ra, beta, alpha, n)
    yout[:] = y + h6 * (dydx + dyt + 2.0 * dym)

def plot_k1_vs_r1(results):
    k1_values_forward = results['k1_values_forward']
    r1_values_forward = results['r1_values_forward']
    k1_values_backward = results['k1_values_backward']
    r1_values_backward = results['r1_values_backward']

    fig, ax = plt.subplots(figsize=(7,6),facecolor='#f8f9fa')

    ax.plot(k1_values_forward, r1_values_forward, 'o--', label='Forward Simulation', markersize=4)
    ax.plot(k1_values_backward, r1_values_backward, 'o--', label='Backward Simulation', markersize=4)

    ax.set_xlabel(r'$K_1$')
    ax.set_ylabel(r'$r_1$')
    ax.set_title(r'$r_1$ v/s $K_1$')
    ax.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return plot_url