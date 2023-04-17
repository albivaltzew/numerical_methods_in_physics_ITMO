from sympy import *
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import inv

# import plotly.io as pio
# print(pio.templates)

A = 14
B = 4
C = 15
D = 6
E = 1
F = 5
# R = 100

# A = 7
# B = 2
# C = 8
# D = 4
# E = 7
# F = 1


x, y = symbols("x y")
fun = 0.5 * A * x ** 2 + B * x * y + 0.5 * C * y ** 2 - D * x - E * y + F

start = np.array([1.0, 1.0], float)


# Analitical

def analitical():
    dfun_dx = diff(fun, x)
    dfun_dy = diff(fun, y)
    analitical_min_func = solve((dfun_dx, dfun_dy), x, y)
    xmin = analitical_min_func.get(x)
    ymin = analitical_min_func.get(y)
    z = fun.subs([(x, xmin), (y, ymin)])
    return xmin, ymin, z


def print_analitical():
    print(f"Analitical: x = {analitical()[0]}, y = {analitical()[1]}, z = {analitical()[2]}")


def check_analitical_solution():
    d2fun_dx2 = diff(fun, x, 2)
    d2fun_dy2 = diff(fun, y, 2)
    d2fun_dxy = diff(fun, x, y)

    xmin, ymin, z = analitical()

    Q = d2fun_dx2.subs((x, xmin), (y, ymin))
    L = d2fun_dy2.subs((x, xmin), (y, ymin))
    M = d2fun_dxy.subs((x, xmin), (y, ymin))
    # Hessian
    m = Matrix([[Q, M],
                [M, L]])
    determinant = m.det()

    if determinant > 0:
        if Q > 0:
            print("There is an extremum and a minimum")
        elif Q < 0:
            print("There is an extremum and a maximum")
    elif determinant < 0:
        print("There is not an extremum")
    elif determinant == 0:
        print("Additional research is required")


def manual_analitical():
    x = (E * B - C * D) / (B ** 2 - C * A)
    y = (D - A * x) / B
    z = 0.5 * A * x ** 2 + B * x * y + 0.5 * C * y ** 2 - D * x - E * y + F
    print(f"\nManual analitical: x = {x}, y = {y}, z = {z}")


# Plot 

def plot_graph():
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)

    def Z(x, y):
        return 0.5 * A * x ** 2 + B * x * y + 0.5 * C * y ** 2 - D * x - E * y + F

    fig = go.Figure(data=[go.Surface(z=Z(X, Y), x=X, y=Y)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Optimization', template="plotly_dark", autosize=False,
                      scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90)
                      )
    fig.show()


# Gradient descent

def gradient(vector):
    grad_x = diff(fun, x).subs([(x, vector[0]), (y, vector[1])])
    grad_y = diff(fun, y).subs([(x, vector[0]), (y, vector[1])])
    return np.array([grad_x, grad_y])


def gradient_descent_method(start, step, n_iter=100, tolerance=1e-06):
    vector = start
    for i in range(n_iter):
        diff = -step * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector = vector + diff
    xmin = vector[0]
    ymin = vector[1]
    z = fun.subs([(x, xmin), (y, ymin)])
    iteration = i
    return xmin, ymin, z, iteration


def print_gradient_descent_method(start, step, n_iter=100, tolerance=1e-06):
    xmin, ymin, z, iteration = gradient_descent_method(start, step, n_iter, tolerance)
    print(f"\nGradient descent method: x = {xmin}, "
          f"y = {ymin}, "
          f"z = {z}"
          f"\nIterations = {iteration}")


# Newton method

def newton_method(start, step, n_iter=100, tolerance=1e-06):
    Q = d2fun_dx2 = diff(fun, x, 2)
    L = d2fun_dy2 = diff(fun, y, 2)
    M = d2fun_dxy = diff(fun, x, y)

    H = np.array(Matrix([[Q, M],
                         [M, L]]), float)
    Hinv = inv(H)
    vector = start
    for iteration in range(1, n_iter):
        Xnew = vector - step * Hinv.dot(gradient(vector))
        if np.all(np.abs(Xnew - vector) <= tolerance):
            break
        vector = Xnew
    xmin = vector[0]
    ymin = vector[1]
    z = fun.subs([(x, xmin), (y, ymin)])
    return xmin, ymin, z, iteration


def print_newton_method(start, step, n_iter=100, tolerance=1e-06):
    xmin, ymin, z, iteration = newton_method(start, step)
    print(f"\nNewton method: x = {xmin}, "
          f"y = {ymin}, "
          f"z = {z}"
          f"\nIterations = {iteration}")


print_analitical()
check_analitical_solution()
print_newton_method(start, 0.65)
print_gradient_descent_method(start, 0.1)
plot_graph()

# manual_analitical()
