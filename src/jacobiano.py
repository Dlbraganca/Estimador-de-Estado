import numpy as np
import math

def Jacob(x, sys, y_bus, med, nbus):  
    ang = nbus - 1
    ten = nbus
    jac = np.empty((0,ang+ten))
    med_types = med[:,1]
    v1 = 0.0
    v2 = 0.0
    teta1 = 0.0
    teta2 = 0.0
    g = 0.0
    b = 0.0j
    g_shunt = 0.0
    b_shunt = 0.0
    #TIPO 1 (FLUXO DE POTêNCIA ATIVA):
    positions = np.where(med_types==1)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 1
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO A busRA PARA   
        data_from = int(med[k][2] - 1)
        data_to = int(med[k][3] - 1)
        if data_from == 0:
            teta1 = 0
            teta2 = x[data_to-1]
        elif data_to == 0:
            teta1 = x[data_from-1]
            teta2 = 0
        else:
            teta1 = x[data_from-1]
            teta2 = x[data_to-1]
        v1 = x[ang + data_from]
        v2 = x[ang + data_to]
        for i in sys:
            if i[0] == data_from+1 and i[1]==data_to+1:
                impedance = i[2] + i[3]
                reactance = 1/impedance
                g = reactance.real
                b = reactance.imag
                break
        if data_from -1 >= 0:
            dteta1 = v1*v2*(g*math.sin(teta1-teta2)-b*math.cos(teta1-teta2))
            value[data_from-1] = dteta1
        if data_to -1 >= 0:
            dteta2 = -1*v1*v2*(g*math.sin(teta1-teta2)-b*math.cos(teta1-teta2))
            value[data_to-1] = dteta2
        dv1 = -v2*(g*math.cos(teta1-teta2)-b*math.sin(teta1-teta2))+2*(g+g_shunt)*v1
        value[data_from+ang] = dv1
        dv2 = -v1*(g*math.cos(teta1-teta2)+b*math.sin(teta1-teta2))
        value[data_to+ang] = dv2
        jac = np.vstack([jac, value])
    
    #TIPO 3 (INJEÇÃO DE POTêNCIA ATIVA):
    positions = np.where(med_types==3)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 3
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO POSICAO DAS busRAS 
        data_from = int(med[k][2] - 1)
        dtetai = 0
        dvi = 0
        #CALCULO DO dP/dTi
        i = data_from
        Bii = y_bus[i][i].imag
        Gii = y_bus[i][i].real
        if i > 0:
            for j in range(nbus):
                if i == 0:
                    teta1 = 0
                else:
                    teta1 = x[i-1]
                if j == 0:
                    teta2 = 0
                else:
                    teta2 = x[j - 1]
                v1 = x[i+(nbus-1)]
                v2 = x[j+(nbus-1)]
                G = y_bus[i][j].real
                B = y_bus[i][j].imag
                dtetai = dtetai + v1*v2*(-G*math.sin(teta1-teta2)+B*math.cos(teta1-teta2))
                dvi = dvi + v2*(G*math.cos(teta1-teta2)+B*math.sin(teta1-teta2))
                if j!=i:
                    if j>0:
                        dtetaj = v1*v2*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                        value[j-1] = dtetaj
                    dvj = v1*(G*math.cos(teta1-teta2)-B*math.sin(teta1-teta2))
                    value[j + nbus - 1] = dvj

            value[i-1] = dtetai - (v1**2)*Bii
            value[i - 1 + nbus] = dvi + v1*Gii
        jac = np.vstack([jac, value])
                
    #TIPO 2 (FLUXO DE POTêNCIA REATIVA):
    positions = np.where(med_types==2)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 1
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO A busRA PARA   
        data_from = int(med[k][2] - 1)
        data_to = int(med[k][3] - 1)
        if data_from == 0:
            teta1 = 0
            teta2 = x[data_to-1]
        elif data_to == 0:
            teta1 = x[data_from-1]
            teta2 = 0
        else:
            teta1 = x[data_from-1]
            teta2 = x[data_to-1]
        v1 = x[ang + data_from]
        v2 = x[ang + data_to]
        for i in sys:
            if i[0] == data_from+1 and i[1]==data_to+1:
                impedance = i[2] + i[3]
                reactance = 1/impedance
                g = reactance.real
                b = reactance.imag
                break
        if data_from -1 >= 0:
            dteta1 = -1*v1*v2*(g*math.cos(teta1-teta2)+b*math.sin(teta1-teta2))
            value[data_from-1] = dteta1
        if data_to -1 >= 0:
            dteta2 = v1*v2*(g*math.cos(teta1-teta2)+b*math.sin(teta1-teta2))
            value[data_to-1] = dteta2
        dv1 = -v2*(g*math.sin(teta1-teta2)-b*math.cos(teta1-teta2))-2*(b+b_shunt)*v1
        value[data_from+ang] = dv1
        dv2 = -v1*(g*math.sin(teta1-teta2)-b*math.cos(teta1-teta2))
        value[data_to+ang] = dv2
        jac = np.vstack([jac, value])
    
    #TIPO 4 (INJEÇÃO DE POTêNCIA REATIVA):
    positions = np.where(med_types==4)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 4
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO POSICAO DAS busRAS 
        data_from = int(med[k][2] - 1)
        dtetai = 0
        dvi = 0
        #CALCULO DO dP/dTi
        i = data_from
        Bii = y_bus[i][i].imag
        Gii = y_bus[i][i].real
        if i > 0:
            for j in range(nbus):
                if i == 0:
                    teta1 = 0
                else:
                    teta1 = x[i-1]
                if j == 0:
                    teta2 = 0
                else:
                    teta2 = x[j - 1]
                v1 = x[i+(nbus-1)]
                v2 = x[j+(nbus-1)]
                G = y_bus[i][j].real
                B = y_bus[i][j].imag
                dtetai = dtetai + v1*v2*(G*math.cos(teta1-teta2)+B*math.sin(teta1-teta2))
                dvi = dvi + v2*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                if j!=i:
                    if j>0:
                        dtetaj = v1*v2*(-G*math.cos(teta1-teta2)-B*math.sin(teta1-teta2))
                        value[j-1] = dtetaj
                    dvj = v1*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                    value[j + nbus - 1] = dvj

            value[i-1] = dtetai - (v1**2)*Gii
            value[i - 1 + nbus] =dvi - v1*Bii
        jac = np.vstack([jac, value])   

    #TIPO 5 (MODULO DE TENSAO):
    positions = np.where(med_types==5)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 4
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO POSICAO DAS busRAS 
        data_from = int(med[k][2] - 1)
        value[ang+data_from] = 1.0
        jac = np.vstack([jac, value]) 
    
    jac = np.where(jac ==-0, 0, jac)
    return np.matrix(jac)
