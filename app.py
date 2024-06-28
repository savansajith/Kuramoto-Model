from flask import Flask, render_template, request
from simulation1 import OscillatorsSimulator, plot_k1_vs_r1

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
