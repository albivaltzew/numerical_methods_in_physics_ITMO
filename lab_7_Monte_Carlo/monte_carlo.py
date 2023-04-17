from scipy import integrate
import math
import numpy as np

A = 2
B = 2
C = 8
R = 1

N = 100000

# Make square domain with uniformly distributed values
x_list = np.random.uniform(-R, R, N)
y_list = np.random.uniform(-R, R, N)


def func(x, y):
    return A * x ** 2 + B * y ** 2 + C


# This is for built in function
def bounds_x():
    return [-R, R]


def bounds_y(x):
    return [-math.sqrt(R ** 2 - x ** 2), math.sqrt(R ** 2 - x ** 2)]


# This is for Monte Carlo
def make_circle(x, y):
    '''
    Make integration domain
    :param x: x
    :param y: y
    :return: True for inside, False for outside
    '''
    return x ** 2 + y ** 2 <= 1


def inner_points(domain):
    # Choose point inside circle
    x_circle = []
    y_circle = []
    for i in range(N):
        if domain(x_list[i], y_list[i]):
            x_circle.append(x_list[i])
            y_circle.append(y_list[i])

    # n - number of inner points
    n = len(x_circle)
    return x_circle, y_circle, n


def monte_carlo_mean(fun, N=1000):
    x_circle, y_circle, n = inner_points(make_circle)
    temp_sum = 0
    for k in range(n):
        temp_sum += fun(x_circle[k], y_circle[k])
    integral_mean = (R - -R) * (R - -R) / N * temp_sum

    return integral_mean


def volume_Monte_Carlo(fun, N=1000):
    # Search max
    x_circle, y_circle, n = inner_points(make_circle)
    fun_value = []
    for k in range(n):
        fun_value.append(fun(x_circle[k], y_circle[k]))
    M = max(fun_value)

    z_list = np.random.uniform(0, M, N)
    counter = 0
    for i in range(N):
        # if area is circle and all points is less than upper surface
        if make_circle(x_list[i], y_list[i]) and z_list[i] < fun(x_list[i], y_list[i]):
            counter += 1
    volume_integral = (R - -R) * (R - -R) * M * counter / N
    return volume_integral


built_in_func = integrate.nquad(func, [bounds_y, bounds_x])[0]

print(f"Calculate the integral using SciPy: {built_in_func}")
print(f"\nCalculate Integral with mean theorem: {monte_carlo_mean(func, N)}")
print(f"\nCalculate Integral as the volume of vertical cylinder: {volume_Monte_Carlo(func, N)}")
