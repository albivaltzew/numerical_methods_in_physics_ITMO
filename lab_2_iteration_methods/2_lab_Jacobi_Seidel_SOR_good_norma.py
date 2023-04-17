import numpy as np
from numpy.linalg import norm, inv
from numpy import dot
from my_gauss import gauss

a = np.array([
    [10, 3, 0],
    [3, 15, 1],
    [0, 1, 7]], float)
b = np.array([2, 12, 5], float)

# a = np.array([
#     [10, 3, 0],
#     [3, 15, 1],
#     [0, 1, 7]], float)
# b = np.array([2, 12, 5], float)
(n,) = np.shape(b)  # array dimensions

A = np.copy(a)
Ainv = inv(A)


# Невязка


def Decompose_matrices(A):
    global Alow, D, Aup, Dinv
    print("\nA = Alow + D + Aup\n")
    Alow = np.copy(A)
    Aup = np.copy(A)

    D = np.diag(np.diag(A))
    Dinv = inv(D)

    for i in range(n):
        for j in range(n):
            if i == j:
                Alow[i][j] = 0
                Aup[i][j] = 0
            elif i > j:
                Alow[i][j] = 0
            elif i < j:
                Aup[i][j] = 0

    print(f"Alow:\n{Alow} \n\nD:\n{D} \n\nAup:\n{Aup}")
    return Alow, D, Aup, Dinv


def Is_diagonal_dominant(A):
    diagonal = np.diag(A)
    abs_diagonal = np.diag(np.abs(A))
    others = np.sum(np.abs(A), axis=1) - diagonal  # np.diag(diagonal) it makes 2-D array
    if np.all(abs_diagonal >= others) and np.all(diagonal > 0):  # axis=1 - sum rows
        print(f"Matrix A is diagonally dominant")
    else:
        print(f"Matrix A is not diagonally dominant")


iterlimit = 100  # number of iteration
tolerance = 1.0e-4  # accuracy degree


def Jacobi():
    x = np.full(n, 1.0, float)
    for iteration in range(1, iterlimit + 1):
        xnew = -Dinv.dot(Alow.dot(x)) - Dinv.dot(Aup.dot(x)) + Dinv.dot(b)
        r = b - a.dot(x)
        if ((norm(A)*norm(Ainv)) * norm(r) / norm(
                b) < tolerance).all():  # if all elements is less than tolerance, all() return True
            print(f"\nJacobi method:\nAnswer: {xnew}\nNumber of iterations: {iteration}")
            break
        else:
            x = np.copy(xnew)


def Seidel():
    x = np.full(n, 1.0, float)
    for iteration in range(1, iterlimit + 1):
        xnew = inv(D + Alow).dot(b - Aup.dot(x))
        r = b - a.dot(x)
        if ((norm(A)*norm(Ainv)) * norm(r) / norm(
                b) < tolerance).all():  # if all elements is less than tolerance, all() return True
            print(f"\nSeidel method:\nAnswer: {xnew}\nNumber of iterations: {iteration}")
            break
        else:
            x = np.copy(xnew)


def Define_omega_SOR():
    omegas = [i for i in np.arange(1.04, 1.06, 0.01)]
    for omega in omegas:
        x = np.full(n, 1.0, float)
        round_omega = round(omega, 2)
        print(f"\nomega:\n{round_omega}")
        for iteration in range(1, iterlimit + 1):
            r = b - a.dot(x)
            xnew = inv(D + Alow * round_omega).dot((b - A.dot(x)) * round_omega) + x
            if ((norm(A)*norm(Ainv)) * norm(r) / norm(
                b) < tolerance).all():  # if all elements is less than tolerance, all() return True
                print(f"\nSOR method:\nAnswer: {xnew}\nNumber of iterations: {iteration}")
                break
            else:
                x = np.copy(xnew)

def SOR(omega=1.04):
    x = np.full(n, 1.0, float)
    for iteration in range(1, iterlimit + 1):
        xnew = inv(D + Alow * omega).dot((b - A.dot(x)) * omega) + x
        r = b - a.dot(x)
        if ((norm(A)*norm(Ainv)) * norm(r) / norm(
                b) < tolerance).all():  # if all elements is less than tolerance, all() return True
            print(
                f"\nSOR method:\nRelaxation parameter \u03C9: {omega}\nAnswer: {xnew}\nNumber of iterations: {iteration}")
            break
        else:
            x = np.copy(xnew)


Is_diagonal_dominant(a)
Decompose_matrices(a)

# print("\nCheck by Gauss: [-0.0296827   0.76560901  0.604913  ]")

Jacobi()
Seidel()
SOR()
# SOR(omega=1.05)

# Define_omega_SOR()

# Check by Gauss
gauss(a, b)
