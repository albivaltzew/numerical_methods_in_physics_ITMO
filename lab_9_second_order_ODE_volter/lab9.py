import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# # From task:
# a1 = 3
# a2 = 2
# c0 = 1
# c1 = 0
# F = lambda x: np.cos(x)

a1 = 0
a2 = 1
c0 = 0
c1 = 1
F = lambda x: 1 + x * 0

f = lambda x: F(x) - c1 * a1 - (c1 * x + c0) * a2
K = lambda x, t: a1 + a2 * (x - t)

x0 = 0
xn = 10
h = 0.01
iterations = 10
N = int((xn - x0) / h)
x = np.linspace(x0, xn, N)
y = F(x)


def calcInt(yTemp):
    yk = yTemp
    for i in range(N):
        sum = 0
        for j in range(i + 1):
            sum += 2 * K(x[j], x[i]) * yTemp[j]
        sum += -K(x[i], x[0]) * yTemp[0] - K(x[i], x[i]) * yTemp[i]
        yk[i] = f(x[i]) + sum * h / 2
    return yk


# yk1 = calcInt(y)
yk = calcInt(y)
for i in range(iterations):
    y = np.copy(yk)
    yk = calcInt(y)

# eps = 1.0e-3
# iter = 0
# while norm(yk - y)/norm(yk) > eps:
#     y = yk
#     yk = calcInt(y)
#     iter += 1
# print(iter)

yTemp = np.copy(yk)
for i in range(N):
    sum = 0
    for j in range(i):
        sum += 2 * (x[i] - x[j]) * yTemp[j]
    sum += -(x[0] - x[i]) * yTemp[0]
    yk[i] = c0 + c1 * x[i] + sum * h / 2


def fun1(x):
    return -np.cos(x) + np.sin(x) + 1


#
#
def fun2(x):
    return (5 / 8) * np.exp(2 * x) - (5 / 8) * np.exp(-2 * x) - 1 / 4


# plt.plot(x, yk1, label="")
# plt.plot(x, fun2(x))
plt.figure()
plt.plot(x, yk, label="Numerical solution")
# plt.plot(x, fun1(x), '--', label="u(x) = sin(x) - cos(x) + 1")
plt.plot(x, fun1(x), '--', label="Exact solution")
plt.legend()
plt.show()
