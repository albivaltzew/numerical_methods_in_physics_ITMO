from numpy import array, zeros, allclose, dot
from numpy.linalg import solve

# a = array([[2, 1, 3],
#            [11, 7, 5],
#            [9, 8, 4]], float)
#
# b = array([10, 2, 6], float)

def gauss(A, B):


    # a = array([
    #     [10, 3, 0],
    #     [3, 15, 1],
    #     [0, 1, 7]], float)
    # b = array([2, 12, 5], float)

    # A = a.copy()
    # B = b.copy()

    n = len(B)

    x = zeros(n, float)

    # 1 step: Elimination
    for k in range(n - 1):
        if A[k, k] == 0:
            for j in range(n):  # swap rows
                A[k, j], A[k + 1, j] = A[k + 1, j], A[k, j]
                B[k], B[k + 1] = B[k + 1], B[k]

        for i in range(k + 1, n):
            if A[i, k] == 0:  # this is save us from nan, division by zero
                continue
            mult = A[i, k] / A[k, k]
            B[i] = B[i] - mult * B[k]
            for j in range(k, n):
                A[i, j] = A[i, j] - mult * A[k, j]

    # 2 step: Back substitution
    x[n - 1] = B[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        terms = 0
        for j in range(i + 1, n):
            terms += A[i, j] * x[j]
        x[i] = (B[i] - terms) / A[i, i]

    print(f"\nThe solution of system by Gauss: \n{x}")


