import numpy as np
import matplotlib.pyplot as plt

x0 = 0
y01 = 1
y02 = 1
xn = 5
step = 1000
x = np.linspace(x0, xn, step + 1)


def dydx(x, y):
    return y


def dydx_2(x, y):
    return -y


def adams_bashfourth_moulton(fun, x0, y0, xn, step):
    # Step size
    h = ((xn - x0) / step)
    y = [y0]
    # function number = fn
    for i in range(step):
        k1 = h * (fun(x0, y0))
        k2 = h * (fun((x0 + h / 2), (y0 + k1 / 2)))
        k3 = h * (fun((x0 + h / 2), (y0 + k2 / 2)))
        k4 = h * (fun((x0 + h), (y0 + k3)))
        yn = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y.append(yn)
        y0 = yn
        x0 = x0 + h

    # Bashfourth:
    y = np.array(y)
    n = step
    for i in range(5, n):
        y[i + 1] = y[i] + (h / 720) * (1901 * fun(x[i], y[i]) -
                                       2774 * fun(x[i - 1], y[i - 1]) +
                                       2616 * fun(x[i - 2], y[i - 2]) -
                                       1274 * fun(x[i - 3], y[i - 3]) +
                                       251 * fun(x[i - 4], y[i - 4])
                                       )
        y[i + 1] = y[i] + (h / 720) * (251 * fun(x[i + 1], y[i + 1]) +
                                       646 * fun(x[i], y[i]) -
                                       264 * fun(x[i - 1], y[i - 1]) +
                                       106 * fun(x[i - 2], y[i - 2]) -
                                       19 * fun(x[i - 3], y[i - 3])
                                       )
    return y


def problem_solution1(x):
    y = np.exp(x)
    return y


def problem_solution2(x):
    y = np.exp(-x)
    return y


def abs_error(calculated_fun, true_fun):
    return abs(calculated_fun - true_fun)


def relative_error(calculated_fun, true_fun):
    # return abs(calculated_fun - true_fun) / calculated_fun
    return abs(calculated_fun - true_fun) / true_fun


def plot_functions(x, fun1, fun2,
                   accurate_sol1, accurate_sol2,
                   abs_error1, relative_error1,
                   abs_error2, relative_error2):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, fun1, label="Adams–Bashforth-Moulton")
    axs[0, 0].plot(x, accurate_sol1, "--", label="y = $e^{x}$",  markersize=0.5)
    axs[0, 0].set_title("y'=y")
    axs[0, 0].legend()

    axs[0, 1].plot(x, fun2, label="Adams–Bashforth-Moulton")
    axs[0, 1].plot(x, accurate_sol2, "--", label="y = $e^{-x}$", markersize=0.5)
    axs[0, 1].set_title("y'=-y")
    axs[0, 1].legend()

    axs[1, 0].plot(x, abs_error1, label="Absolute error")
    axs[1, 0].plot(x, relative_error1, "--", label="Relative error", markersize=0.5)
    axs[1, 0].set_title("")
    axs[1, 0].legend()

    axs[1, 1].plot(x, abs_error2, label="Absolute error")
    axs[1, 1].plot(x, relative_error2, "--", label="Relative error", markersize=0.5)
    axs[1, 1].set_title("")
    axs[1, 1].legend()
    plt.show()


plot_functions(x, adams_bashfourth_moulton(dydx, x0, y01, xn, step),
               adams_bashfourth_moulton(dydx_2, x0, y02, xn, step),
               problem_solution1(x), problem_solution2(x),
               abs_error(adams_bashfourth_moulton(dydx, x0, y01, xn, step), problem_solution1(x)),
               relative_error(adams_bashfourth_moulton(dydx, x0, y01, xn, step), problem_solution1(x)),
               abs_error(adams_bashfourth_moulton(dydx_2, x0, y02, xn, step), problem_solution2(x)),
               relative_error(adams_bashfourth_moulton(dydx_2, x0, y02, xn, step), problem_solution2(x))
               )
