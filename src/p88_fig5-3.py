import numpy as np

x = np.array(
    [
        [2, -1, -1, 0, 0],
        [-1, 3, -1, -1, 0],
        [-1, -1, 3, -1, 0],
        [0, -1, -1, 3, -1],
        [0, 0, 0, -1, 1],
    ]
)

w, v = np.linalg.eig(x)

v = v[:, np.argsort(w)]
print(np.argsort(w))
print(v)