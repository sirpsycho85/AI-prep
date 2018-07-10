import numpy as np
from Fully_Connected import Fully_Connected

num_classes = 10
num_features = 2
x = np.array([1, 0.1])
fc = Fully_Connected(num_classes, num_features)
fc_out = fc.forward(x)
ds = np.random.random(num_classes)
dw = fc.backward(ds)
fc.update(0.1)
