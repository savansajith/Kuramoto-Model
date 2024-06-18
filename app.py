from flask import Flask,render_template
from sympy import *

# Define symbols
i, j, k, K, N, t = symbols('i j k K N t')
K1 = Symbol("K1")
K2 = Symbol("K2")
theta_i = Function("theta_i")
theta_j = Function("theta_j")
theta_k = Function("theta_k")
omega_i = Symbol("omega_i")

# Define equations
expr1 = Eq(diff(theta_i(t), t), omega_i + (K / N) * Sum(sin(theta_j(t) - theta_i(t)), (j, 1, N)))
expr2 = Eq(diff(theta_i(t), t), omega_i + (K1 / N) * Sum(sin(-theta_i(t) + theta_j(t)), (i, 1, N)) + (K2 / N**2) * Sum(Sum(sin(2 * theta_j(t) - theta_k(t) - theta_i(t)), (j, 1, N)), (k, 1, N)))

# Convert equations to LaTeX format
latex_expr1 = latex(expr1)
latex_expr2 = latex(expr2)

Eqs_lst = [
    {
             "order": "First Order:",
             "eq": latex_expr1
    },
    {
             "order": "Second Order:",
             "eq": latex_expr2
    }
]

app = Flask(__name__)

@app.route('/')
def hello_savan():
    return render_template("home.html", eqs=Eqs_lst)

if __name__ == '__main__':
         app.run(host="0.0.0.0", debug=True)