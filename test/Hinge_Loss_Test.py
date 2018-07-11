import numpy as np
from Hinge_Loss import Hinge_Loss

scores = np.array([5, 2.7, 4.5, 4.1, 0.1, 5.0])
hinge = Hinge_Loss()
loss = hinge.forward(scores, 0)
ds = hinge.backward(1)
print(loss)  # ~1.6
print(ds)  # [3. 0. 1. 1. 0. 1.]
