from flask import Flask, render_template, request
from simulation1 import OscillatorsSimulator, plot_k1_vs_r1
from simulation2 import simulate_and_plot
import io
import base64

app = Flask(__name__)

# Home route, renders the home.html template with a LaTeX equation
@app.route('/')
def home():
    # LaTeX equation for display on the webpage
    latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
    return render_template("home.html", eq=latex_eq)

# Route to handle plotting based on form submission
@app.route('/plot', methods=['POST'])
def plot():
    try:
        # Retrieve form data from POST request
        k1_start = float(request.form['k1_start'])
        k1_end = float(request.form['k1_end'])
        k2 = float(request.form['k2'])
        n = int(request.form['n'])
        tran = int(request.form['tran'])
        niter = int(request.form['niter'])
        dk = float(request.form['dk'])

        # Parameters for simulation
        h = 0.01
        # Initialize the OscillatorsSimulator object
        simulator = OscillatorsSimulator(k1_start, k1_end, k2, n, tran, niter, h, dk)
        # Run simulation
        results = simulator.simulate()
        # Generate plot URL
        plot_url = plot_k1_vs_r1(results)

        # LaTeX equation for display on the webpage
        latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
        return render_template("home.html", eq=latex_eq, plot_url=plot_url)
    except Exception as e:
        # Error handling for any exceptions during plotting
        return f"An error occurred: {e}\nPlease recheck your entered values."

# Route to handle second simulation and plotting based on form submission
@app.route('/second_simulation', methods=['POST'])
def second_simulation():
    try:
        # Retrieve form data from POST request
        n = int(request.form['n2'])
        k1 = float(request.form['k1_2'])
        k2 = float(request.form['k2_2'])

        # Perform second simulation and generate plot
        fig, ax = simulate_and_plot(n, k1, k2)

        # Save the plot as a base64 encoded string for display in HTML
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        second_plot_url = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # LaTeX equation for display on the webpage
        latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"
        return render_template("home.html", eq=latex_eq, second_plot_url=second_plot_url)
    except Exception as e:
        # Error handling for any exceptions during second simulation and plotting
        return f"An error occurred: {e}\nPlease recheck your entered values."

if __name__ == '__main__':
    # Run the Flask application
    app.run(host="0.0.0.0", debug=True)
