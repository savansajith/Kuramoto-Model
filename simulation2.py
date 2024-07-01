# theta vs time code

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Parameters and Constants
tran = 2000
nite = 3000
h = 0.05
pi = np.pi

@njit
def initialize(n):
    omega = np.zeros(n)
    theta = np.random.uniform(-pi, pi, n)
    for i in range(n):
        omega[i] = np.tan((i * pi) / float(n) - ((n + 1) * pi) / float(2 * n))
    return omega, theta

@njit
def compute_derivs(t, theta, k1, k2, omega, r1, shi1, r2, shi2):
    dth1 = k1 * r1 * np.sin(shi1 - theta)
    dth2 = k2 * r2 * r1 * np.sin(shi2 - shi1 - theta)
    dth = omega + dth1 + dth2
    return dth

@njit
def rk4(t, theta, h, k1, k2, omega, r1, shi1, r2, shi2):
    hh = h * 0.5
    h6 = h / 6.0
    k1_derivs = compute_derivs(t, theta, k1, k2, omega, r1, shi1, r2, shi2)
    yt = theta + hh * k1_derivs
    k2_derivs = compute_derivs(t + hh, yt, k1, k2, omega, r1, shi1, r2, shi2)
    yt = theta + hh * k2_derivs
    k3_derivs = compute_derivs(t + hh, yt, k1, k2, omega, r1, shi1, r2, shi2)
    yt = theta + h * k3_derivs
    k4_derivs = compute_derivs(t + h, yt, k1, k2, omega, r1, shi1, r2, shi2)
    theta_new = theta + h6 * (k1_derivs + 2.0 * (k2_derivs + k3_derivs) + k4_derivs)
    return theta_new

def simulate_and_plot(n, k1, k2):
    omega, theta = initialize(n)
    r = 0.0
    r_2 = 0.0
    t = 0.0

    for it in range(1, nite + 1):
        rx1 = np.sum(np.cos(theta))
        ry1 = np.sum(np.sin(theta))
        rx2 = np.sum(np.cos(2 * theta))
        ry2 = np.sum(np.sin(2 * theta))
        r1 = np.sqrt(rx1**2 + ry1**2) / float(n)
        r2 = np.sqrt(rx2**2 + ry2**2) / float(n)
        shi1 = np.arctan2(ry1, rx1)
        shi2 = np.arctan2(ry2, rx2)
        if it > tran:
            r += r1
            r_2 += r2
        theta = rk4(t, theta, h, k1, k2, omega, r1, shi1, r2, shi2)
        t += h

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
    ax.plot(theta, np.ones_like(theta), 'o', label='Oscillator', color='k')
    ax.set_title(f"Oscillators at t = {t:.2f} seconds")
    ax.set_rticks([0, 1])
    ax.legend()
    ax.grid(True)

    return fig, ax

if __name__ == "__main__":
    n = int(input("Enter the number of oscillators (N): "))
    k2 = float(input("Enter the value of K_2: "))
    k1 = float(input("Enter the value of K_1: "))

    simulate_and_plot(n, k1, k2)
