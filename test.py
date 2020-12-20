import numpy as np

a = np.ones(3).reshape(3, 1)
b = np.zeros(3).reshape(3, 1)
c = np.ones(3).reshape(3, 1)
d = np.append(np.append(a, b, 1), c, 1)

print(d)