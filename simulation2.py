import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def simulate_and_animate(n, k1, k2):
    omega = np.random.rand(n) * 2 * np.pi - np.pi
    theta = np.random.rand(n) * 2 * np.pi - np.pi

    fig, ax = plt.subplots()
    line, = ax.plot(np.cos(theta), np.sin(theta), 'ro')

    def update(num, theta, line):
        rs1 = np.sin(theta).sum()
        rc1 = np.cos(theta).sum()
        rs2 = np.sin(2 * theta).sum()
        rc2 = np.cos(2 * theta).sum()
        dtheta = omega + (k1 / n) * (np.cos(theta) * rs1 - np.sin(theta) * rc1) + (k2 / n**2) * (np.cos(2 * theta) * rs2 - np.sin(2 * theta) * rc2)
        theta += dtheta * 0.01
        theta = np.mod(theta, 2 * np.pi)
        line.set_data(np.cos(theta), np.sin(theta))
        return line,

    ani = animation.FuncAnimation(fig, update, frames=200, fargs=(theta, line), interval=20, blit=True)
    return ani
