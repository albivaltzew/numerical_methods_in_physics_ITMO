import numpy as np
from numpy.linalg import cholesky
from numpy import transpose, dot

a = np.array([[81, -45, 45],
              [-45, 50, -15],
              [45, -15, 38]], dtype=np.double)

n = len(a)

U = a.copy()
L = np.zeros((n, n), dtype=float)

for i in range(n):
    for k in range(i + 1):
        tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
        if i == k:
            L[i, i] = np.sqrt(U[i, i] - tmp_sum)
        else:
            L[i, k] = (1 / L[k, k]) * (U[i, k] - tmp_sum)



print(f"\nThe solution of system: \n{L}")
print(f"\nA = LL^T: \n{dot(L, transpose(L))}")
print("\n-----------check-------------")
l_chol = cholesky(a)
print(f"\nCheck by built in functions: \n{l_chol}")
