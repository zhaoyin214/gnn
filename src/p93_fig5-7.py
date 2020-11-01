import numpy as np

A = np.array(
    [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ]
)

x = np.array([1, 0, 0, -1, -1])
h = [1, 0.5, 0.5]

x1 = np.matmul(A, x)
x2 = np.matmul(A, x1)
y = np.sum([hk * xk for hk, xk in zip(h, [x, x1, x2])], axis=0)

print(x1)
print(x2)
print(y)