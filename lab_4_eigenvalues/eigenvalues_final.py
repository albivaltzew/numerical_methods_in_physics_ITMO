from sympy import *
import numpy as np
from numpy.linalg import eig, norm

a = np.array([[16, 3, 2],
              [3, 5, 1],
              [2, 1, 10]], float)
# a = np.array([[5, 8, 3],
#               [8, 7, 4],
#               [3, 4, 19]], float)

n = len(a)
b = np.zeros(n, dtype=float)
tolerance = 1.0e-5
tau = 0.5


def Get_characteristic_equation():
    # make lamda variable and real
    global lamda, characteristic_equation
    lamda = var("lamda", real=True)
    # Generate matrix with ones on main diagonal
    I_matrix = eye(n)
    IL = I_matrix * lamda
    A_IL = a - IL
    characteristic_equation = A_IL.det()
    return characteristic_equation, A_IL


def Print_characteristic_equation():
    characteristic_equation, _ = Get_characteristic_equation()
    print(f"\n|a - lamda * I|:\n{characteristic_equation} = 0 \n")


def Newton_method():
    characteristic_equation, _ = Get_characteristic_equation()
    eigenvalue = [10 * i for i in range(n)]
    eigenvaluenew = [1] * n
    eigenvalues_float = [1] * n
    # find derivative for Newton method
    derivative_char_equation = diff(characteristic_equation, lamda)
    for i in range(n):
        for iteration in range(1, 101):
            eigenvaluenew[i] = eigenvalue[i] - \
                               characteristic_equation.subs(lamda, eigenvalue[i]) / \
                               derivative_char_equation.subs(lamda, eigenvalue[i])
            if abs(eigenvaluenew[i] - eigenvalue[i]) < tolerance:
                eigenvalues_float[i] = N(eigenvalue[i])
                break
            eigenvalue[i] = eigenvaluenew[i]
    return eigenvalues_float


def Print_Eigenvalues_Newton_method():
    eigenvalues = Newton_method()
    print(f"Eigenvalues by Newton method: {eigenvalues}\n")


# k - number eigenvalue
def Substitute_values(k):
    _, A_IL = Get_characteristic_equation()
    eigenvalue = Newton_method()
    substituted_values_sympy = A_IL.subs(lamda, eigenvalue[k])
    substituted_values = np.array(substituted_values_sympy).astype(np.float64)
    return substituted_values


A = np.array(a, dtype='float64')
Aorig = np.copy(A)


def Error(eigenVector, eigenValue):
    global Aorig
    return np.linalg.norm(((Aorig - (np.eye(n) * eigenValue)).dot(eigenVector)).astype('float64'))


Print_Eigenvalues_Newton_method()
f = np.array([[0] for _ in range(len(A))])

eigenValues = Newton_method()

for k in range(n):
    eigenValue = eigenValues[k]
    print(f"Eigenvalue {k + 1}: {eigenValue}")
    A = Substitute_values(k)
    A = np.array(A, dtype="float64")
    print(A)
    Xstart = np.full(n, 1.0, float)


    def Jacobi():
        global XJacobi, iterationJacobi
        XJacobi = np.copy(Xstart)
        iterationJacobi = 0
        while Error(XJacobi, eigenValue) > tolerance:
            xnew = np.full(n, 1.0, float)
            for i in range(len(A)):
                sum = 0
                for j in range(len(A)):
                    sum += A[i, j] * XJacobi[j]
                xnew[i] = XJacobi[i] + (b[i] - sum) / A[i, i] * tau
            XJacobi = xnew
            iterationJacobi += 1
        return XJacobi


    def Seidel():
        global XSeidel, iterationSeidel
        XSeidel = np.copy(Xstart)
        iterationSeidel = 0
        while Error(XSeidel, eigenValue) > tolerance:  # Seidel iter
            xnew = [1 for _ in range(len(A))]
            for i in range(len(A)):
                sum1 = 0
                for j in range(i):
                    sum1 += A[i, j] * xnew[j]
                sum2 = 0
                for j in range(i + 1, len(A)):
                    sum2 += A[i, j] * XSeidel[j]
                xnew[i] = (b[i] - sum1 - sum2) / A[i, i]
            XSeidel = xnew
            iterationSeidel += 1
        return XSeidel


    def SOR():
        global XSOR, iterationSOR
        w = 1
        XSOR = np.copy(Xstart)
        iterationSOR = 0
        while Error(XSOR, eigenValue) > tolerance:
            xnew = [1 for _ in range(len(A))]
            for i in range(len(A)):
                sum1 = 0
                for j in range(i):
                    sum1 += A[i, j] * xnew[j]
                sum2 = 0
                for j in range(i + 1, len(A)):
                    sum2 += A[i, j] * XSOR[j]
                xnew[i] = w * (b[i] - sum1 - sum2) / A[i, i] + (1 - w) * XSOR[i]
            XSOR = xnew
            iterationSOR += 1
        return XSOR


    coefJacobi = norm(Jacobi())
    coefSeidel = norm(Seidel())
    coefSOR = norm(SOR())

    for i in range(n):
        XJacobi[i] = XJacobi[i] / coefJacobi
        XSeidel[i] = XSeidel[i] / coefSeidel
        XSOR[i] = XSOR[i] / coefSOR

    print(f"Jacobi's method: {XJacobi}\nIterations: {iterationJacobi}\n\nSeidel's method:"
          f" {XSeidel}\nIterations: {iterationSeidel}\n\nSOR method: {XSOR}\nIterations: {iterationSOR}")
    print('--------------------------------------------')


def eigenvalues_by_sympy_solver():
    roots = solveset(characteristic_equation, lamda)
    real_roots = simplify(roots)
    print(f"\nEigenvalues by sympy solver: \n{real_roots.evalf()}")


def Check_value_by_built_in_functions():
    value, vectors = eig(a)
    print("\nCheck by built in functions:")
    print(f"\nEigen values: \n{value}\nEigen vectors: \n{vectors}")


Check_value_by_built_in_functions()
