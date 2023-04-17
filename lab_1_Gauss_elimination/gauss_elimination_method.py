from numpy import array, zeros, allclose, dot
from numpy.linalg import solve

a = array([[2, 1, 3],
           [11, 7, 5],
           [9, 8, 4]], float)

b = array([10, 2, 6], float)

# a = array([
#     [3, -1, 0, 2],
#     [-2, 1, 1, 2],
#     [0, -1, 7, 2],
#     [-1, 2, 3, 5]], float)
# b = array([2, 1, -3, 2], float)

A = a.copy()
B = b.copy()

n = len(b)

x = zeros(n, float)

# 1 step: Elimination
for k in range(n - 1):
    if a[k, k] == 0:
        for j in range(n):  # swap rows
            a[k, j], a[k + 1, j] = a[k + 1, j], a[k, j]
            b[k], b[k + 1] = b[k + 1], b[k]

    for i in range(k + 1, n):
        if a[i, k] == 0:  # this is save us from nan, division by zero
            continue
        mult = a[i, k] / a[k, k]
        b[i] = b[i] - mult * b[k]
        for j in range(k, n):
            a[i, j] = a[i, j] - mult * a[k, j]
print(a)
# 2 step: Back substitution
x[n - 1] = b[n - 1] / a[n - 1, n - 1]
for i in range(n - 2, -1, -1):
    terms = 0
    for j in range(i + 1, n):
        terms += a[i, j] * x[j]
    x[i] = (b[i] - terms) / a[i, i]

print(f"\nThe solution of system: \n{x}")
print("\n-----------check-------------")
X = solve(A, B)
print(f"\nCheck by built in functions: \n{X}")
print("\nCheck AX = B: ")
print(allclose(dot(A, X), B))
