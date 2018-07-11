import numpy as np
from Hinge_Loss import Hinge_Loss

scores = np.array([5, 2.7, 4.5, 4.1, 0.1, 4.9])
hinge = Hinge_Loss()
predicted_class, loss = hinge.forward(scores, 0)
ds = hinge.backward(1)
print(predicted_class)  # 0
print(loss)  # ~1.5
print(ds)  # [-3. 0. 1. 1. 0. 1.]
