import numpy as np
from PIL import Image, ImageDraw
from numba import njit
import tempfile
import os
import random

# Parameters and Constants
tran = 2000
nite = 3000
h = 0.05
pi = np.pi

# Function to initialize parameters and arrays
@njit
def initialize(n):
    omega = np.zeros(n)
    theta = np.random.uniform(-pi, pi, n)

    for i in range(n):
        omega[i] = np.tan((i * pi) / float(n) - ((n + 1) * pi) / float(2 * n))

    return omega, theta

# Function to compute derivatives
@njit
def compute_derivs(t, theta, k1, k2, omega, r1, shi1, r2, shi2):
    dth1 = k1 * r1 * np.sin(shi1 - theta)
    dth2 = k2 * r2 * r1 * np.sin(shi2 - shi1 - theta)
    dth = omega + dth1 + dth2
    return dth

# Function to perform Runge-Kutta integration
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

# Main simulation function with animation using Pillow
def simulate_and_animate(n, k1, k2):
    omega, theta = initialize(n)
    r = 0.0
    r_2 = 0.0
    t = 0.0

    frames = []

    # Assign random colors to each oscillator
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(n)]

    for it in range(1, nite + 1):
        if it % 10 == 0:  # Save every 10th frame to reduce file size
            img = Image.new('RGB', (600, 600), 'white')
            draw = ImageDraw.Draw(img)

            for i in range(n):
                x = 300 + 200 * np.cos(theta[i])
                y = 300 + 200 * np.sin(theta[i])
                draw.ellipse((x-5, y-5, x+5, y+5), fill=colors[i])

            frames.append(img)

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

    r /= float(nite - tran)
    r_2 /= float(nite - tran)

    return frames

# Function to save the animation
def simulate_and_save_animation(n, k1, k2):
    frames = simulate_and_animate(n, k1, k2)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
        frames[0].save(tmpfile.name, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop = 0)
        with open(tmpfile.name, 'rb') as f:
            gif_data = f.read()
    os.remove(tmpfile.name)
    return gif_data