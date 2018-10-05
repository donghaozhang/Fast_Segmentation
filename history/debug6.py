import numpy as np
x = np.array([1, 2])
print(x.shape)
y = np.expand_dims(x, axis=0)
print(y.shape)