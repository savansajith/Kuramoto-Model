import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

# Parameters and Constants
tran = 2000  # Transient duration
nite = 1000  # Number of iterations for simulation
h = 0.05  # Time step
pi = np.pi  # Value of pi

@njit
def initialize(n):
    """
    Initialize the phases and natural frequencies of oscillators.

    Parameters:
    n (int): Number of oscillators.

    Returns:
    tuple: omega (natural frequencies), theta (initial phases).
    """
    omega = np.zeros(n)
    theta = np.random.uniform(-pi, pi, n)

    # Calculate natural frequencies
    for i in range(n):
        omega[i] = np.tan((i * pi) / float(n) - ((n + 1) * pi) / float(2 * n))

    return omega, theta

@njit
def compute_derivs(t, theta, k1, k2, omega, r1, shi1, r2, shi2):
    """
    Compute the derivatives of the phases.

    Parameters:
    t (float): Time.
    theta (numpy array): Phases of oscillators.
    k1 (float): Coupling strength for pairwise interactions.
    k2 (float): Coupling strength for higher-order interactions.
    omega (numpy array): Natural frequencies.
    r1 (float): Order parameter for pairwise interactions.
    shi1 (float): Phase coherence for pairwise interactions.
    r2 (float): Order parameter for higher-order interactions.
    shi2 (float): Phase coherence for higher-order interactions.

    Returns:
    numpy array: Derivatives of the phases.
    """
    dth1 = k1 * r1 * np.sin(shi1 - theta)
    dth2 = k2 * r2 * r1 * np.sin(shi2 - shi1 - theta)
    dth = omega + dth1 + dth2
    return dth

@njit
def rk4(t, theta, h, k1, k2, omega, r1, shi1, r2, shi2):
    """
    Perform fourth-order Runge-Kutta integration to update phases.

    Parameters:
    t (float): Time.
    theta (numpy array): Phases of oscillators.
    h (float): Time step.
    k1 (float): Coupling strength for pairwise interactions.
    k2 (float): Coupling strength for higher-order interactions.
    omega (numpy array): Natural frequencies.
    r1 (float): Order parameter for pairwise interactions.
    shi1 (float): Phase coherence for pairwise interactions.
    r2 (float): Order parameter for higher-order interactions.
    shi2 (float): Phase coherence for higher-order interactions.

    Returns:
    numpy array: Updated phases after integration.
    """
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

def simulate_and_animate(n, k1, k2):
    """
    Simulate and animate the Kuramoto model.

    Parameters:
    n (int): Number of oscillators.
    k1 (float): Coupling strength for pairwise interactions.
    k2 (float): Coupling strength for higher-order interactions.

    Returns:
    matplotlib.animation.FuncAnimation: Animation object.
    """
    omega, theta = initialize(n)
    t = 0.0

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'bo', markersize=5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    def update(it):
        """
        Update function for animation.

        Parameters:
        it (int): Frame index.

        Returns:
        tuple: Updated plot elements.
        """
        nonlocal theta, t
        rx1 = np.sum(np.cos(theta))
        ry1 = np.sum(np.sin(theta))
        rx2 = np.sum(np.cos(2 * theta))
        ry2 = np.sum(np.sin(2 * theta))

        r1 = np.sqrt(rx1**2 + ry1**2) / float(n)
        r2 = np.sqrt(rx2**2 + ry2**2) / float(n)

        shi1 = np.arctan2(ry1, rx1)
        shi2 = np.arctan2(ry2, rx2)

        theta = rk4(t, theta, h, k1, k2, omega, r1, shi1, r2, shi2)
        t += h

        x = np.cos(theta)
        y = np.sin(theta)
        line.set_data(x, y)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=nite, blit=True, interval=30)  # Animation object
    return ani

# Example usage
if __name__ == "__main__":
    n = int(input("Enter the number of oscillators (N): "))
    k2 = float(input("Enter the value of K_2: "))
    k1 = float(input("Enter the value of K_1: "))

    ani = simulate_and_animate(n, k1, k2)
    ani.save("kuramoto_simulation.gif", writer='imagemagick', fps=30)