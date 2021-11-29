import numpy as np
import ch01.forward_net as fn

if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = fn.TwoLayerNet(2, 4, 3)
    s = model.predict(x)

    print(s)
