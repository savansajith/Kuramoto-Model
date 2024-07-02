# r1 vs k1 code

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from concurrent.futures import ProcessPoolExecutor
import numba as nb

# Define OscillatorsSimulator class for simulation
class OscillatorsSimulator:
    def __init__(self, k1_start, k1_end, k2, n, tran, niter, h, dk):
        # Initialize simulation parameters
        self.k1_start = k1_start
        self.k1_end = k1_end
        self.k2 = k2
        self.n = n
        self.tran = tran
        self.niter = niter
        self.h = h
        self.dk = dk

    # Method to perform simulations and return results
    def simulate(self):
        # Set up initial conditions and random state
        pi = np.arctan(1.0) * 4
        random_state = np.random.RandomState(1234568)
        omega = np.tan((np.arange(self.n) * pi) / self.n - ((self.n + 1) * pi) / (2 * self.n))
        theta = -1.0 * pi + 2.0 * pi * random_state.rand(self.n)

        # Run forward and backward simulations
        forward_results = self.run_simulation(theta, omega, self.k1_start, self.k1_end, self.dk, self.n, self.tran, self.niter, self.h, self.k2)
        theta = 2 * pi * np.ones(self.n)
        backward_results = self.run_simulation(theta, omega, self.k1_end, self.k1_start, -self.dk, self.n, self.tran, self.niter, self.h, self.k2)

        # Return simulation results
        return {
            'k1_values_forward': forward_results[0],
            'r1_values_forward': forward_results[1],
            'k1_values_backward': backward_results[0],
            'r1_values_backward': backward_results[1],
        }

    # Method to run the simulation process
    def run_simulation(self, theta, omega, k1_start, k1_end, dk, n, tran, niter, h, k2):
        k1_values = []
        r1_values = []

        # Use ProcessPoolExecutor for parallel simulation runs
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for K1 in np.arange(k1_start, k1_end + np.sign(dk) * 0.01, dk):
                futures.append(executor.submit(run_simulation_step, theta.copy(), omega, K1, k2, n, tran, niter, h))

            for future in futures:
                K1, r1 = future.result()
                k1_values.append(K1)
                r1_values.append(r1)

        return k1_values, r1_values

# Decorated function for numerical simulation step using numba
@nb.njit
def run_simulation_step(theta, omega, K1, K2, n, tran, niter, h):
    pi = np.arctan(1.0) * 4
    r1 = 0.0
    for it in range(1, niter + 1):
        # Calculate various sums and averages for the simulation
        rc1 = np.cos(theta).sum()
        rs1 = np.sin(theta).sum()
        rc2 = np.cos(2 * theta).sum()
        rs2 = np.sin(2 * theta).sum()
        ra = np.sqrt(rs1 ** 2 + rc1 ** 2) / n
        dth = np.zeros_like(theta)
        # Perform derivative calculations
        derivs(0, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
        tho = np.zeros_like(theta)
        rk4(theta, dth, n, 0, h, tho, omega, K1, K2, ra, rs1, rs2, rc1, rc2)
        theta = np.mod(tho, 2 * pi)
        if it > tran:
            r1 += ra
    r1 /= niter - tran
    return K1, r1

# Decorated function for derivative calculations using numba
@nb.njit
def derivs(t, dth, theta, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n):
    for i in range(n):
        # Calculate derivatives
        dth[i] = omega[i] + (K1 / n) * (np.cos(theta[i]) * rs1 - np.sin(theta[i]) * rc1) + \
                 (K2 / n ** 2) * (np.cos(theta[i]) * rs2 * rc1 - np.sin(theta[i]) * rs2 * rs1 -
                                 np.sin(theta[i]) * rc2 * rc1 - np.cos(theta[i]) * rc2 * rs1)

# Decorated function for Runge-Kutta method using numba
@nb.njit
def rk4(y, dydx, n, x, h, yout, omega, K1, K2, ra, rs1, rs2, rc1, rc2):
    dym = np.zeros_like(y)
    dyt = np.zeros_like(y)
    yt = np.zeros_like(y)
    hh = h * 0.5
    h6 = h / 6.0
    xh = x + hh
    yt = y + hh * dydx
    derivs(xh, dyt, yt, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    yt = y + hh * dyt
    derivs(xh, dym, yt, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    yt = y + h * dym
    derivs(x + h, dyt, yt, omega, K1, K2, rs1, rs2, rc1, rc2, ra, n)
    yout[:] = y + h6 * (dydx + dyt + 2.0 * dym)

# Function to plot k1 vs r1 and return plot as base64 encoded URL
def plot_k1_vs_r1(results):
    k1_values_forward = results['k1_values_forward']
    r1_values_forward = results['r1_values_forward']
    k1_values_backward = results['k1_values_backward']
    r1_values_backward = results['r1_values_backward']

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k1_values_forward, r1_values_forward, 'o-', label='Forward Simulation', markersize=4)
    ax.plot(k1_values_backward, r1_values_backward, 'o-', label='Backward Simulation', markersize=4)

    # Set plot labels and title
    ax.set_xlabel(r'$K_1$')
    ax.set_ylabel(r'$r_1$')
    ax.set_title(r'$r_1$ v/s $K_1$')
    ax.legend()
    plt.grid()

    # Convert plot to base64 encoded URL
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return plot_url
