from numpy import array, eye, dot

a = array([[2, 1, 3],
           [11, 7, 5],
           [9, 8, 4]], float)
# a = array([
#     [3, -1, 0, 2],
#     [-2, 1, 1, 2],
#     [0, -1, 7, 2],
#     [-1, 2, 3, 5]], float)

# a = array([[11, 8, 3],
#            [8, 10, 5],
#            [3, 5, 12]], float)

n = len(a)
U = a.copy()
L = eye(n, dtype=float)

for k in range(n - 1):  # choose current line
    for i in range(k + 1, n):  # for the elements of the first column, excluding the element of the main row
        mult = U[i, k] / U[k, k]  # determine the multiplier for this string
        L[i, k] = mult  # adding the multiplier to the lower matrix
        for j in range(n):  # we run through all the elements of this line
            U[i, j] = U[i, j] - mult * U[k, j]  # We multiply each element of the main row by a multiplier
                                                # and subtract it from the modified row

print(f"\nU matrix: \n\n{U}\n")
print(f"L matrix: \n\n{L}\n")
print(f"A = LU: \n{dot(L, U)}")
