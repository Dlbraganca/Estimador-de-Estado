from turtle import position
from src.buscar_ramos import find_branch
from src.buscar_angulo import get_teta
from src.buscar_v import get_v
import numpy as np
import math

def Jacob(x, sys, y_bus, med, nbus):  
    x  = np.array(x)[:,0]
    ang = nbus - 1
    ten = nbus
    jac = np.empty((0,ang+ten))
    med_types = med[:,1]
    v1 = 0.0
    v2 = 0.0
    teta1 = 0.0
    teta2 = 0.0
    g = 0.0
    b = 0.0
    g_shunt = 0.0
    b_shunt = 0.0
    for k in range(np.shape(med_types)[0]):
    #TIPO 1 (FLUXO DE POTêNCIA ATIVA):
        if(med_types[k] == 1):
            value = np.zeros(ang+ten)

            #DEFININDO AS BARRAS E POSICOES NO JACOBIANO  
            data_from = int(med[k][2])
            position_from = data_from - 1
            data_to = int(med[k][3])
            position_to = data_to - 1

            #RECEBENDO OS VALORES DE V TETA E RAMOS
            v1, v2 = get_v(x, data_from, data_to, nbus)
            teta1, teta2 = get_teta(x, data_from, data_to)
            g, b, g_shunt, b_shunt = find_branch(sys, data_from, data_to)

            #CALCULANDOAS DERIVADAS DE TETA
            if data_from != 1:
                dteta1 = v1*v2*(g*math.sin(teta1 - teta2) - b*math.cos(teta1 - teta2))
                value[position_from - 1] = dteta1
            if data_to != 1:
                dteta2 = -1*v1*v2*(g*math.sin(teta1 - teta2) - b*math.cos(teta1 - teta2))
                value[position_to - 1] = dteta2
            
            #CALCULANDO AS DERIVADAS DE TENSÃO
            dv1 = -v2*(g*math.cos(teta1-teta2)+b*math.sin(teta1-teta2))+2*(g+g_shunt)*v1
            value[position_from + ang] = dv1
            dv2 = -v1*(g*math.cos(teta1-teta2)+b*math.sin(teta1-teta2))
            value[position_to + ang] = dv2
            
            #ADICIONA NO JACOBIANO
            jac = np.vstack([jac, value])

        if(med_types[k] == 3):       
        #TIPO 3 (INJEÇÃO DE POTêNCIA ATIVA):
            value = np.zeros(ang+ten)

            #DEFININDO AS BARRAS E POSICOES NO JACOBIANO  
            data_from = int(med[k][2])
            position_from = data_from - 1
            data_to = int(med[k][3])
            position_to = data_to - 1
            dteta1 = 0
            dv1 = 0
            Gii = y_bus[position_from][position_from].real
            Bii = y_bus[position_from][position_from].imag
            for j in range(nbus):

                data_to = j + 1
                position_to = j
                #RECEBENDO OS VALORES DE V TETA E RAMOS
                v1, v2 = get_v(x, data_from, data_to, nbus)
                teta1, teta2 = get_teta(x, data_from, data_to)
                G = y_bus[position_from][position_to].real
                B = y_bus[position_from][position_to].imag
                
                #CALCULANDO OS SOMATORIOS
                dteta1 = dteta1 + v1*v2*(-G*math.sin(teta1-teta2)+B*math.cos(teta1-teta2))
                dv1 = dv1 + v2*(G*math.cos(teta1-teta2)+B*math.sin(teta1-teta2))

                #CALCULADO A DERIVADA DE TETA 2
                if data_to != 1:
                    dteta2 = v1*v2*(G*math.sin(teta1 - teta2) - B*math.cos(teta1 - teta2))
                    value[position_to - 1] = dteta2

                #CALCULADO A DERIVADA DE v 2
                dv2 = v1*(G*math.cos(teta1 - teta2) + B*math.sin(teta1 - teta2))
                value[position_to + ang] = dv2

            #DEFININDO VALORES DAS DERIVADAS DE TETA E V
            if data_to != 1:
                value[position_from - 1] = dteta1 - (v1**2)*Bii
            value[position_from + ang] = dv1 + v1*Gii
            
            #ADICIONA NO JACOBIANO
            jac = np.vstack([jac, value])
        
        if(med_types[k] == 2):
            #TIPO 2 (FLUXO DE POTêNCIA REATIVA):
            value = np.zeros(ang+ten)

            #DEFININDO AS BARRAS E POSICOES NO JACOBIANO  
            data_from = int(med[k][2])
            position_from = data_from - 1
            data_to = int(med[k][3])
            position_to = data_to - 1

            #RECEBENDO OS VALORES DE V TETA E RAMOS
            v1, v2 = get_v(x, data_from, data_to, nbus)
            teta1, teta2 = get_teta(x, data_from, data_to)
            g, b, g_shunt, b_shunt = find_branch(sys, data_from, data_to)

            #CALCULANDOAS DERIVADAS DE TETA
            if data_from != 1:
                dteta1 = -v1*v2*(g*math.cos(teta1 - teta2) + b*math.sin(teta1 - teta2))
                value[position_from - 1] = dteta1
            if data_to != 1:
                dteta2 = v1*v2*(g*math.cos(teta1 - teta2) + b*math.sin(teta1 - teta2))
                value[position_to - 1] = dteta2
            
            #CALCULANDO AS DERIVADAS DE TENSÃO
            dv1 = -v2*(g*math.sin(teta1 - teta2) - b*math.cos(teta1 - teta2)) - 2*v1*(b+b_shunt)
            value[position_from + ang] = dv1
            dv2 = -v1*(g*math.sin(teta1 - teta2) - b*math.cos(teta1 - teta2))
            value[position_to + ang] = dv2
            
            #ADICIONA NO JACOBIANO
            jac = np.vstack([jac, value])
    
        if(med_types[k] == 4):
        #TIPO 4 (INJEÇÃO DE POTêNCIA REATIVA):
            value = np.zeros(ang+ten)

            #DEFININDO AS BARRAS E POSICOES NO JACOBIANO  
            data_from = int(med[k][2])
            position_from = data_from - 1
            data_to = int(med[k][3])
            position_to = data_to - 1
            dteta1 = 0
            dv1 = 0
            Gii = y_bus[position_from][position_from].real
            Bii = y_bus[position_from][position_from].imag
            for j in range(nbus):

                data_to = j + 1
                position_to = j
                #RECEBENDO OS VALORES DE V TETA E RAMOS
                v1, v2 = get_v(x, data_from, data_to, nbus)
                teta1, teta2 = get_teta(x, data_from, data_to)
                G = y_bus[position_from][position_to].real
                B = y_bus[position_from][position_to].imag
                
                #CALCULANDO OS SOMATORIOS
                dteta1 = dteta1 + v1*v2*(G*math.cos(teta1-teta2) + B*math.sin(teta1-teta2))
                dv1 = dv1 + v2*(G*math.sin(teta1-teta2) - B*math.cos(teta1-teta2))

                #CALCULADO A DERIVADA DE TETA 2
                if data_to != 1:
                    dteta2 = v1*v2*(-G*math.cos(teta1 - teta2) - B*math.sin(teta1 - teta2))
                    value[position_to - 1] = dteta2

                #CALCULADO A DERIVADA DE v 2
                dv2 = v1*(G*math.sin(teta1 - teta2) - B*math.cos(teta1 - teta2))
                value[position_to + ang] = dv2

            #DEFININDO VALORES DAS DERIVADAS DE TETA E V
            if data_from != 1:
                value[position_from - 1] = dteta1 - (v1**2)*Gii
            value[position_from + ang] = dv1 - v1*Bii
            
            #ADICIONA NO JACOBIANO
            jac = np.vstack([jac, value])  

        if(med_types[k] == 5):
            #TIPO 5 (MODULO DE TENSAO):

            value = np.zeros(ang+ten)
            #ENCONTRANDO POSICAO DAS BARRAS 
            data_from = int(med[k][2])
            position_from = data_from - 1

            #DEFININDO VALOR DA DERIVADA
            value[position_from + ang] = 1.0

            #ADICIONANDO NO JACOBIANO
            jac = np.vstack([jac, value]) 
    
    #SUBSTITUINDO OS -0 POR 0
    jac = np.where(jac ==-0, 0, jac)
    return np.matrix(jac, dtype=np.float64)
    #return jac
