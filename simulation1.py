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
    for it in range(1, niter + 1):
        rc1 = np.cos(theta).sum()
        rs1 = np.sin(theta).sum()
        rc2 = np.cos(2 * theta).sum()
        rs2 = np.sin(2 * theta).sum()
        ra = np.sqrt(rs1 ** 2 + rc1 ** 2) / n
        dth = np.zeros_like(theta)
        derivs(0, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
        tho = np.zeros_like(theta)
        rk4(theta, dth, n, 0, h, tho, omega, K1, K2, ra, rs1, rs2, rc1, rc2)
        theta = np.mod(tho, 2 * pi)
        if it > tran:
            r1 += ra
    r1 /= niter - tran
    return K1, r1

@nb.njit
def derivs(t, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n):
    for i in range(n):
        dth[i] = omega[i] + (K1 / n) * (np.cos(theta[i]) * rs1 - np.sin(theta[i]) * rc1) + (K2 / n ** 2) * (np.cos(2 * theta[i]) * rs2 - np.sin(2 * theta[i]) * rc2)

@nb.njit
def rk4(th, dth, n, t, h, tho, omega, K1, K2, ra, rs1, rs2, rc1, rc2):
    th1, th2, th3 = th.copy(), th.copy(), th.copy()
    k1 = dth.copy()
    for i in range(n):
        th1[i] = th[i] + k1[i] * h / 2.0
    rc1 = np.cos(th1).sum()
    rs1 = np.sin(th1).sum()
    rc2 = np.cos(2 * th1).sum()
    rs2 = np.sin(2 * th1).sum()
    derivs(t + h / 2.0, k1, th1, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    k2 = k1.copy()
    for i in range(n):
        th2[i] = th[i] + k2[i] * h / 2.0
    rc1 = np.cos(th2).sum()
    rs1 = np.sin(th2).sum()
    rc2 = np.cos(2 * th2).sum()
    rs2 = np.sin(2 * th2).sum()
    derivs(t + h / 2.0, k2, th2, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    k3 = k2.copy()
    for i in range(n):
        th3[i] = th[i] + k3[i] * h
    rc1 = np.cos(th3).sum()
    rs1 = np.sin(th3).sum()
    rc2 = np.cos(2 * th3).sum()
    rs2 = np.sin(2 * th3).sum()
    derivs(t + h, k3, th3, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    for i in range(n):
        tho[i] = th[i] + h * (dth[i] + 2.0 * (k1[i] + k2[i]) + k3[i]) / 6.0

def plot_k1_vs_r1(results):
    fig, ax = plt.subplots()
    ax.plot(results['k1_values_forward'], results['r1_values_forward'], label='Forward')
    ax.plot(results['k1_values_backward'], results['r1_values_backward'], label='Backward')
    ax.set_xlabel('K1')
    ax.set_ylabel('R1')
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_url
