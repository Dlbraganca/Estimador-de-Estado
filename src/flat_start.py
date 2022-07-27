import numpy as np

def Flat_start(nBus):
    x = []
    # x = np.array([])
    for i in range(2*nBus - 1):
        if i+1 > nBus-1:
            # x = np.append(x, 1.0)
            x.append(1.0)
        else:
            # x = np.append(x, 0.0)
            x.append(0.0)
    x = np.matrix(x, dtype=np.float64).transpose()
    return x
