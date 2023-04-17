import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy.polynomial.chebyshev import chebinterpolate

A1 = 4
A2 = 2
w1 = 3
w2 = 5

N = 16

t = np.linspace(-np.pi, 0, 64)
x = np.cos(t)


def f(x):
    return A1 * np.cos(w1 * x) + A2 * np.sin(w2 * x)


def f_a0(t):
    return A1 * np.cos(w1 * np.cos(t)) + A2 * np.sin(w2 * np.cos(t))


def f_a(t, n=1):
    return (A1 * np.cos(w1 * np.cos(t)) + A2 * np.sin(w2 * np.cos(t))) * np.cos(n * t)


def coef_a0():
    integral_a0, _ = integrate.quad(f_a0, 0, np.pi)
    a0 = (1 / np.pi) * integral_a0
    return a0


def coef_a(n):
    integral_a, _ = integrate.quad(f_a, 0, np.pi, args=(n,))
    a = (2 / np.pi) * integral_a
    return a


def Cheb(t, n=N):
    a0 = coef_a0()
    sum = np.zeros(np.size(t))
    sum += a0 * np.cos(0 * t)
    for i in range(1, n + 1):
        sum += coef_a(i) * np.cos(i * t)
    return sum


def Error_value(fun1, fun2):
    return abs(fun1 - fun2)


# Plot Graphs
def plot_functions(X, fun1, fun2):
    plt.title("Chebyshev approximation")
    plt.plot(X, fun1, label="f(x)")
    plt.plot(X, fun2, ".r", label="$S^{(2)}_{N}$(x)")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    # plt.text(-0.9, 4, f'N = {n}')
    # plt.axis([-3, 3, -6.5, 6])
    plt.axis([x[0], x[-1], -6.5, 6])


def plot_Error_function(X, fun1):
    plt.plot(X, fun1, label="|f(x) - $S^{(2)}_{N}$(x)|")
    plt.xlabel("x")
    plt.grid(True)
    plt.legend()
    # plt.axis([-3.01, 3.01, 0, 7])
    plt.axis([X[0], X[-1], 0, 7 * 10 ** (-8)])


def Plot_this_graphs(X, fun1, fun2):
    plt.figure(figsize=(7, 7))
    plt.subplot(211)
    plot_functions(X, fun1, fun2)
    # plt.subplots_adjust(hspace=0.5)
    plt.subplot(212)
    plot_Error_function(X, Error_value(fun1, fun2))
    plt.show()


# Print coeff a
def print_coef_a(n=N):
    print(f"Coefficient  a0 : {coef_a0()}")
    for i in range(0, n):
        print(f"\n{[j for j in range(1, n + 1)][i]} coefficient a: {coef_a(i)}")


# запись в файл
def write_coef_in_file(n=N):
    f = open("a coefficients_integration.txt", "w", encoding="utf-8")
    f.write(f"Coefficient a0 : {coef_a0()}")
    for i in range(1, n):
        f.write(f"\n{[j for j in range(0, n + 1)][i]} coefficient a: {coef_a(i)}")
    f.close()


write_coef_in_file(N)
print_coef_a()

print("Built in Function")
print(np.real(chebinterpolate(f, N)))


Plot_this_graphs(x, f(x), Cheb(t, int(N)))