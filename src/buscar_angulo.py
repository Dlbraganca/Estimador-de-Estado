import numpy as np

def get_teta(x, busFrom, busTo):
    
    teta1 = 0.0
    teta2 = 0.0
    if busFrom == 1:
        teta1 = 0.0
    else:
        teta1 = x[busFrom - 2].item()
    if busTo == 1:
        teta2 = 0.0
    else:
        teta2 = x[busTo - 2].item()

    return teta1, teta2