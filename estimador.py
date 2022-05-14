from ast import expr_context
from cgi import print_environ
import math
import numpy as np
from src.jacobiano import Jacob
from src.y_barra import Y_bus
from src.matriz_covariancia import Cov_m
from src.flat_start import Flat_start
from src.h_vetor import h

if __name__ == "__main__":


    
    #DEFININE PARAMETROS DE PRECISAO
    #np.set_printoptions(precision=2, suppress=True)

    #SISTEMA
    #[DE,PARA,RESISTÊNCIA,REATâNCIA. ADMITANCIA SHUNT,SUSCEPTÂNCIA SHUNT]
    # sys = np.array([[1,2,0.00,0.40j,0, 0j],
    #                 [1,3,0.00,0.20j,0, 0j],
    #                 [2,3,0.00,0.25j,0, 0j],
    #                 [2,4,0.00,0.50j,0, 0j],
    #                 [3,4,0.00,0.25j,0, 0j]])

    sys = np.array([[1,2,0.01,0.03j,0, 0j],
                    [1,3,0.02,0.05j,0, 0j],
                    [2,3,0.03,0.08j,0, 0j]])

    tolerance = 0.0001
    end_iteration = False


    #MEDIÇÕES
    #[NÚMERO, TIPO, DE, PARA, VALOR, DESVIO PADRÃO]
    #1-FLUXO DE POTÊNCIA ATIVA
    #2-FLUXO DE POTÊNCIA REATIVA
    #3-INJEÇÃO DE POTÊNCIA ATIVA
    #4-INJEÇÃO DE POTêNCIA REATIVA
    #5-MÓDULO DE TENSÃO

    # med = np.array([[1,1,1,2,0.70,0.0333],
    #                 [2,2,1,2,0.15,0.0333],
    #                 [3,1,1,3,-0.35,0.0333],
    #                 [4,2,1,3,-0.10,0.0333],
    #                 [5,3,2,0,1.40,0.0333],
    #                 [6,4,2,0,0.85,0.0333],
    #                 [7,3,4,0,-0.30,0.0333],
    #                 [8,4,4,0,-0.25,0.0333],
    #                 [9,5,1,0,1.02,0.0033],
    #                 [10,5,4,0,0.99,0.0033]
    #                 ])

    med = np.array([[1,1,1,2,0.888,0.008],
                    [2,1,1,3,1.173,0.008],
                    [3,3,2,0,-0.501,0.010],
                    [4,2,1,2,0.568,0.008],
                    [5,2,1,3,0.663,0.008],
                    [6,4,2,0,-0.286,0.010],
                    [7,5,1,0,1.006,0.004],
                    [8,5,2,0,0.968,0.004]])

    print(sys)
    print(med)

    y_bus = Y_bus(sys)

    print(y_bus)

    z = np.matrix(med[:,4]).transpose()

    max = 0
    for i in sys:
        if i[0] > max:
            max = i[0]
        elif i[1] > max:
            max = i[1]
    nbus = int(max.real)
    x = Flat_start(nbus)
    R = Cov_m(med)
    Rinv = np.linalg.inv(R)
    countIt = 0
    while not end_iteration:
        countIt = countIt + 1
        print('-------------- %d ITERAÇÃO--------' %countIt)
        H = Jacob(x, sys, y_bus, med, nbus)
        print(H)
        Htrans = H.transpose()
        hx = h(x, sys, med, nbus, y_bus)
        print(hx)
        G = Htrans*Rinv*H
        #TESTE DE OBSERVABILIDADE
        try:
            Ginv = np.linalg.inv(G)
        except:
            print("SISTEMA NÃO OBSERVÁVALE")
            end_iteration = True
        else:
            fx = Htrans*Rinv
            fx = fx*(z - hx)
            dx = Ginv*fx
            print(G)
            print(fx)
            print(dx)
            #dx = np.multiply(Ginv, np.multiply(Htrans, np.multiply(Rinv, (np.subtract(z, hx)))))
            if abs(np.max(dx)) <= tolerance:
                end_iteration = True
                x = x + dx
                print(x)
                print(h(x, sys, med, nbus, y_bus))
            else:
                x = x + dx
                print(dx)
                print(x)
            

