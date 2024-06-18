from flask import Flask,render_template
from sympy import *

# Define symbols
i, j, k, K, N, t = symbols('i j k K N t')
K_1 = Symbol("K_1")
K_2 = Symbol("K_2")
theta_i = Function("theta_i")
theta_j = Function("theta_j")
theta_k = Function("theta_k")
omega_i = Symbol("omega_i")

# Define equations
expr1_order = Eq(diff(theta_i(t), t), omega_i + (K / N) * Sum(sin(theta_j(t) - theta_i(t)), (j, 1, N)))
expr2_order = Eq(diff(theta_i(t), t), omega_i + (K_1 / N) * Sum(sin(-theta_i(t) + theta_j(t)), (i, 1, N)) + (K_2 / N**2) * Sum(Sum(sin(2 * theta_j(t) - theta_k(t) - theta_i(t)), (j, 1, N)), (k, 1, N)))

# Convert equations to LaTeX format
latex_eq1 = latex(expr1_order)
latex_eq2 = latex(expr2_order)

latex_theta_i = latex(theta_i)
latex_theta_j = latex(theta_j)
latex_theta_k = latex(theta_k)
latex_omega_i = latex(omega_i)
latex_K = latex(K)
latex_K1 = latex(K_1)
latex_K2 = latex(K_2)
latex_N = latex(N)


Eqs_lst = [
    {
             "order": "First Order :",
             "eq": latex_eq1
    },
    {
             "order": "Second Order :",
             "eq": latex_eq2
    }
]

app = Flask(__name__)

@app.route('/')
def hello_savan():
    return render_template("home.html", eqs=Eqs_lst)

if __name__ == '__main__':
         app.run(host="0.0.0.0", debug=True)