from flask import Flask, render_template

app = Flask(__name__)

# LaTeX equation for the second order Kuramoto model
latex_eq = r"\frac{d\theta_i}{dt} = \omega_i + \frac{K_1}{N} \sum_{j=1}^{N} \sin\left(\theta_j - \theta_i\right) + \frac{K_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\left(2\theta_j - \theta_k - \theta_i\right)"

@app.route('/')
def home():
    return render_template("home.html", eq=latex_eq)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
