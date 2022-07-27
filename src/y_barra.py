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
    for i in range(np.shape(y_buss)[0]):
        for k in range(np.shape(y_buss)[1]):
            value = 0 + 0j
            if k < i:
                for l in range(len(sys)):
                    if (sys[l][0] == i+1 and sys[l][1] == k+1) or (sys[l][0] == k+1 and sys[l][1] == i+1):
                        a = i
                        b = k
                        # #VERSÃO Y
                        # y_buss[a][b] =  -1*np.complex(sys[l][2] + sys[l][3]*1j)
                        # y_buss[b][a] =  -1*np.complex(sys[l][2] + sys[l][3]*1j)
                        #VERSÃO Z
                        y_buss[a][b] =  -1/np.complex(sys[l][2] + sys[l][3]*1j)
                        y_buss[b][a] =  -1/np.complex(sys[l][2] + sys[l][3]*1j)
            elif i == k:
                for l in range(len(sys)):
                    if sys[l][0] == i+1 or sys[l][1] == i+1:
                        #VERSÃO Y
                        value = value + 1/np.complex(sys[l][2] + sys[l][3]*1j) + np.complex(sys[l][4] + sys[l][5]*1j)/2
                        # #VERSÃO Z
                        # value = value + 1/np.complex(sys[l][2] + sys[l][3]*1j) + 1/(np.complex(sys[l][4] + sys[l][5]*1j)/2)
                y_buss[i][k] = value
    return y_buss