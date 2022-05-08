import numpy as np

def Y_bus(sys):
#def dimensão do y busra
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
    return y_buss