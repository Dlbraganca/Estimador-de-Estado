import math
import numpy as np

def Y_bus(sys):
#def dimensão do y barra
    max = 0
    for i in sys:
        if i[0] > max:
            max = i[0]
        elif i[1] > max:
            max = i[1]
##################    
    y_buss = np.zeros((int(max.real),int(max.real)), dtype=complex)
    for i in range(y_buss.shape[0]):
        value = 0 + 0j
        for k in range(y_buss.shape[1]):
            if k < i:
                for j in range(sys.shape[0]):
                    if (sys[j][0] == i+1 and sys[j][1] == k+1) or (sys[j][0] == k+1 and sys[j][1] == i+1):
                        a = i
                        b = k
                        y_buss[a][b] =  -1/(sys[j][2] + sys[j][3])
                        y_buss[b][a] =  -1/(sys[j][2] + sys[j][3])
            elif i == k:
                for j in range(sys.shape[0]):
                    if sys[j][0] == i+1 or sys[j][1] == i+1:
                        value = value + 1/(sys[j][2] + sys[j][3])
                y_buss[i][k] = value
    print(y_buss)
    return y_buss

def Flat_start(size):
    x = np.zeros(size*2-1,dtype='float')
    for i in range(x.shape[0]):
        if i+1 > size-1:
            x[i] = 1
    return x

def Jacob(x, sys, y_bus, med, size):  
    ang = size - 1
    ten = size
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
        #ENCONTRANDO A BARRA PARA   
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
        #ENCONTRANDO POSICAO DAS BARRAS 
        data_from = int(med[k][2] - 1)
        dtetai = 0
        dvi = 0
        #CALCULO DO dP/dTi
        i = data_from
        Bii = y_bus[i][i].imag
        Gii = y_bus[i][i].real
        if i > 0:
            for j in range(size):
                if i == 0:
                    teta1 = 0
                else:
                    teta1 = x[i-1]
                if j == 0:
                    teta2 = 0
                else:
                    teta2 = x[j - 1]
                v1 = x[i+(size-1)]
                v2 = x[j+(size-1)]
                G = y_bus[i][j].real
                B = y_bus[i][j].imag
                dtetai = dtetai + v1*v2*(-G*math.sin(teta1-teta2)+B*math.cos(teta1-teta2))
                dvi = dvi + v2*(G*math.cos(teta1-teta2)+B*math.sin(teta1-teta2))
                if j!=i:
                    if j>0:
                        dtetaj = v1*v2*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                        value[j-1] = dtetaj
                    dvj = v1*(G*math.cos(teta1-teta2)-B*math.sin(teta1-teta2))
                    value[j + size - 1] = dvj

            value[i-1] = dtetai - (v1**2)*Bii
            value[i - 1 + size] = dvi + v1*Gii
        jac = np.vstack([jac, value])
                
    #TIPO 2 (FLUXO DE POTêNCIA REATIVA):
    positions = np.where(med_types==2)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 1
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO A BARRA PARA   
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
        #ENCONTRANDO POSICAO DAS BARRAS 
        data_from = int(med[k][2] - 1)
        dtetai = 0
        dvi = 0
        #CALCULO DO dP/dTi
        i = data_from
        Bii = y_bus[i][i].imag
        Gii = y_bus[i][i].real
        if i > 0:
            for j in range(size):
                if i == 0:
                    teta1 = 0
                else:
                    teta1 = x[i-1]
                if j == 0:
                    teta2 = 0
                else:
                    teta2 = x[j - 1]
                v1 = x[i+(size-1)]
                v2 = x[j+(size-1)]
                G = y_bus[i][j].real
                B = y_bus[i][j].imag
                dtetai = dtetai + v1*v2*(G*math.cos(teta1-teta2)+B*math.sin(teta1-teta2))
                dvi = dvi + v2*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                if j!=i:
                    if j>0:
                        dtetaj = v1*v2*(-G*math.cos(teta1-teta2)-B*math.sin(teta1-teta2))
                        value[j-1] = dtetaj
                    dvj = v1*(G*math.sin(teta1-teta2)-B*math.cos(teta1-teta2))
                    value[j + size - 1] = dvj

            value[i-1] = dtetai - (v1**2)*Gii
            value[i - 1 + size] =dvi - v1*Bii
        jac = np.vstack([jac, value])   

    #TIPO 5 (MODULO DE TENSAO):
    positions = np.where(med_types==5)[0]
    #ITERAÇÂO PARA A QUANTIDADE DE MEDIDAS TIPO 4
    for k in positions:
        value = np.zeros(ang+ten)
        #ENCONTRANDO POSICAO DAS BARRAS 
        data_from = int(med[k][2] - 1)
        value[ang+data_from] = 1.0
        jac = np.vstack([jac, value]) 
    
    return jac

def Cov_m(med):
    r_matrix = np.zeros((med.shape[0],med.shape[0]),dtype=float)
    count = 0
    med_types = med[:,1]
    positions = np.where(med_types==1)[0]
    for k in positions:
#        value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][5]**(2)
        #r_matrix = np.append(r_matrix, value)
        count = count + 1
    positions = np.where(med_types==3)[0]
    for k in positions:
#        value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][5]**(2)
        #r_matrix = np.append(r_matrix, value)
        count = count + 1
    positions = np.where(med_types==2)[0]
    for k in positions:
      #  value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][5]**(2)
        #r_matrix = np.append(r_matrix, value)
        count = count + 1
    positions = np.where(med_types==4)[0]
    for k in positions:
#        value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][5]**(2)
        #r_matrix = np.append(r_matrix, value)
        count = count + 1
    positions = np.where(med_types==5)[0]
    for k in positions:
    #   value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][5]**(2)
        #r_matrix = np.append(r_matrix, value)
        count = count + 1

    return r_matrix

#SISTEMA
#[DE,PARA,RESISTÊNCIA,REATâNCIA,SUSCEPTÂNCIA]
sys = np.array([[1,2,0.01,0.03j,0],
                [1,3,0.02,0.05j,0],
                [2,3,0.03,0.08j,0]])

#MEDIÇÕES
#[NÚMERO, TIPO, DE, PARA, VALOR, DESVIO PADRÃO]
#1-FLUXO DE POTÊNCIA ATIVA
#2-FLUXO DE POTÊNCIA REATIVA
#3-INJEÇÃO DE POTÊNCIA ATIVA
#4-INJEÇÃO DE POTêNCIA REATIVA
#5-MÓDULO DE TENSÃO

med = np.array([[1,1,1,2,0.888,0.008],
                [2,1,1,3,1.173,0.008],
                [3,3,2,0,-0.501,0.010],
                [4,2,1,2,0.568,0.008],
                [5,2,1,3,0.663,0.008],
                [6,4,2,0,-0.286,0.010],
                [7,5,1,0,1.006,0.004],
                [8,5,2,0,0.968,0.004]])

#print(sys)
#print(med)
y_bus = Y_bus(sys)

#flat start
#def dimensão do y barra
max = 0
for i in sys:
    if i[0] > max:
        max = i[0]
    elif i[1] > max:
        max = i[1]
size = int(max.real)
x = Flat_start(size)
H = Jacob(x, sys, y_bus, med, size)
H = np.matrix(H)
R = Cov_m(med)
R = np.matrix(R)
print('-----------Y BARRA-------------------------')
print(np.matrix(y_bus))
print('----------------------------------')
print('---------------H--------------------')
print(H)
print('-----------------------------')
print('---------------R------------------')
print(R)
print('------------------------------------')
G = H.transpose()*np.linalg.inv(R)*H
print('-----------------G----------------------')
print(np.matrix(G))
print('-----------------------------------')