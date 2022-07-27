from xmlrpc.client import METHOD_NOT_FOUND
import numpy as np
import math
from src.buscar_v import get_v
from src.buscar_angulo import get_teta
from src.buscar_ramos import find_branch

def h(x, sys, med, nBus, yBus):
    h = np.array([])   
    med_types = med[:,1]
    for k in range(np.shape(med_types)[0]):
     #TIPO 1 (FLUXO DE POTêNCIA ATIVA):
        if(med_types[k] == 1):
            busFrom = int(med[k][2])
            busTo = int(med[k][3])
            v1, v2 = get_v(x, busFrom, busTo, nBus)
            teta1, teta2 = get_teta(x, busFrom, busTo)
            g, b, gs, _ = find_branch(sys, busFrom, busTo)
            
            if g != None:
                value = v1**2*(gs+g) - v1*v2*(g*math.cos(teta1 - teta2)+b*math.sin(teta1-teta2))
                h = np.append(h, value)
        
        if(med_types[k] == 3):
        #TIPO 3 (INJEÇÃO DE POTêNCIA ATIVA):
            busFrom = int(med[k][2])
            value = 0.0

            for i in range(nBus):
                busTo = i+1
                v1, v2 = get_v(x, busFrom, busTo, nBus)
                teta1, teta2 = get_teta(x, busFrom, busTo)
                G = yBus[busFrom - 1][busTo - 1].real
                B = yBus[busFrom - 1][busTo - 1].imag

                value = value + v1*v2*(G*math.cos(teta1 - teta2)+B*math.sin(teta1-teta2))

            h = np.append(h, value)    

        if(med_types[k] == 2):
        #TIPO 2 (FLUXO DE POTêNCIA REATIVA):
            busFrom = int(med[k][2])
            busTo = int(med[k][3])
            v1, v2 = get_v(x, busFrom, busTo, nBus)
            teta1, teta2 = get_teta(x, busFrom, busTo)
            g, b, gs, bs = find_branch(sys, busFrom, busTo)
            
            if g != None:
                value = -v1**2*(bs+b) - v1*v2*(g*math.sin(teta1 - teta2) - b*math.cos(teta1-teta2))
                h = np.append(h, value)

        if(med_types[k] == 4):
        #TIPO 4 (INJEÇÃO DE POTêNCIA REATIVA):
            busFrom = int(med[k][2])
            value = 0.0

            for i in range(nBus):
                busTo = i+1
                
                v1, v2 = get_v(x, busFrom, busTo, nBus)
                teta1, teta2 = get_teta(x, busFrom, busTo)
                G = yBus[busFrom - 1][busTo - 1].real
                B = yBus[busFrom - 1][busTo - 1].imag
                
                value = value + v1*v2*(G*math.sin(teta1 - teta2) - B*math.cos(teta1-teta2))

            h = np.append(h, value)    

        if(med_types[k] == 5):
        #TIPO 5 (MÓDULO DE TENSÃO):
            value = 0.0
            busFrom = int(med[k][2])
            value, _ = get_v(x, busFrom, 0, nBus)
            h = np.append(h, value)
    
    return np.matrix(h, dtype=np.float64).transpose()
