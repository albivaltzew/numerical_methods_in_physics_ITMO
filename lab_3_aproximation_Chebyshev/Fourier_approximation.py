import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

A1 = 4
A2 = 2
w1 = 3
w2 = 5

N = 16

# Define x, 1000 - accuracy
x = np.linspace(-1, 1, 64)


def f(x):
    return A1 * np.cos(w1 * x) + A2 * np.sin(w2 * x)


def f_coefa(x, n=0):
    return (A1 * np.cos(w1 * x) + A2 * np.sin(w2 * x)) * np.cos(np.pi * n * x)


def f_coefb(x, n=0):
    return (A1 * np.cos(w1 * x) + A2 * np.sin(w2 * x)) * np.sin(np.pi * n * x)


# Fourier
# Lets calculate coeff a
def coef_a(n):
    integral_a, error = integrate.quad(f_coefa, -1, 1, args=(n,))
    a = integral_a
    return a


def coef_b(n):
    integral_b, error = integrate.quad(f_coefb, -1, 1, args=(n,))
    b = integral_b
    return b


def Fourier(x, n=N):
    a0 = coef_a(0)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        sum += coef_a(i) * np.cos(i * np.pi * x) + coef_b(i) * np.sin(i * np.pi * x)
    return a0 / 2 + sum


def Error_value(x, fun1, fun2):
    return abs(fun1 - fun2)

def plot_functions(X, fun1, fun2):
    plt.title("Fourier approximation")
    plt.plot(X, fun1, label="f(x)")
    plt.plot(X, fun2, ".r", label="$S^{(1)}_{N}$(x)")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    # plt.text(-0.9, 4, 'f(x), $S^{(1)}_{N}$(x)')
    plt.axis([-1.01, 1.01, -6.5, 6])


def plot_Error_function(X, fun1):
    # plt.title("Fourier approximation")
    plt.plot(X, fun1, label="|f(x) - $S^{(1)}_{N}$(x)|")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    plt.axis([-1.01, 1.01, 0, 2])


def Plot_this_graphs():
    plt.figure(figsize=(7, 7))
    plt.subplot(211)
    plot_functions(x, f(x), Fourier(x, N))
    # plt.subplots_adjust(hspace=0.5)
    plt.subplot(212)
    plot_Error_function(x, Error_value(x, f(x), Fourier(x, N)))
    plt.show()

Plot_this_graphs()
