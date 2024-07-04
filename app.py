from flask import Flask, render_template, request
from simulation1 import OscillatorsSimulator, plot_k1_vs_r1
from simulation2 import simulate_and_animate
import io
import base64
import matplotlib.pyplot as plt
import os
import tempfile
import concurrent.futures

app = Flask(__name__)

@app.route('/')
def home():
    latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
    return render_template("home.html", eq=latex_eq)

@app.route('/plot', methods=['POST'])
def plot():
    try:
        k1_start = float(request.form['k1_start'])
        k1_end = float(request.form['k1_end'])
        k2 = float(request.form['k2'])
        n = int(request.form['n'])
        tran = int(request.form['tran'])
        niter = int(request.form['niter'])
        dk = float(request.form['dk'])

        h = 0.01
        simulator = OscillatorsSimulator(k1_start, k1_end, k2, n, tran, niter, h, dk)
        results = simulator.simulate()
        plot_url = plot_k1_vs_r1(results)

        latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
        return render_template("home.html", eq=latex_eq, plot_url=plot_url)
    except Exception as e:
        return f"An error occurred: {e}\nPlease recheck your entered values."

@app.route('/second_simulation', methods=['POST'])
def second_simulation():
    try:
        n = int(request.form['n2'])
        k1 = float(request.form['k1_2'])
        k2 = float(request.form['k2_2'])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(simulate_and_save_animation, n, k1, k2)
            gif_data = future.result()

        second_plot_url = base64.b64encode(gif_data).decode('utf-8')

        latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
        return render_template("home.html", eq=latex_eq, second_plot_url=second_plot_url)
    except Exception as e:
        return f"An error occurred: {e}\nPlease recheck your entered values."

def simulate_and_save_animation(n, k1, k2):
    ani = simulate_and_animate(n, k1, k2)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
        ani.save(tmpfile.name, writer='imagemagick', fps=30)
        tmpfile.seek(0)
        gif_data = tmpfile.read()
    os.remove(tmpfile.name)
    return gif_data

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
