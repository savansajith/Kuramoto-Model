from flask import Flask, render_template, request
from simulation1 import OscillatorsSimulator, plot_k1_vs_r1
from simulation2 import simulate_and_save_animation
import base64

app = Flask(__name__)

@app.route('/')
def home():
    latex_eq1 = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
    latex_eq2 = r"r_1 e^{i\psi_1} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}"
    return render_template("home.html", latex_eq1=latex_eq1, latex_eq2=latex_eq2)

@app.route('/simulation')
def simulation():
    return render_template("simulation.html")

@app.route('/plot', methods=['POST'])
def plot():
    try:
        k1_start = float(request.form['k1_start'])
        k1_end = float(request.form['k1_end'])
        k2 = float(request.form['k2'])
        n = int(request.form['n'])
        dk = float(request.form['dk'])

        h = 0.01
        tran = 20000
        niter = 30000
        simulator = OscillatorsSimulator(k1_start, k1_end, k2, n, tran, niter, h, dk)
        results = simulator.simulate()
        plot_url = plot_k1_vs_r1(results)

        return render_template("simulation.html", plot_url=plot_url)
    except Exception as e:
        return f"An error occurred: {e}\nPlease recheck your entered values."

@app.route('/second_simulation', methods=['POST'])
def second_simulation():
    try:
        n = int(request.form['n2'])
        k1 = float(request.form['k1_2'])
        k2 = float(request.form['k2_2'])

        gif_data = simulate_and_save_animation(n, k1, k2)

        second_plot_url = base64.b64encode(gif_data).decode('utf-8')

        return render_template("simulation.html", second_plot_url=second_plot_url)
    except Exception as e:
        return f"An error occurred: {e}\nPlease recheck your entered values."

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)