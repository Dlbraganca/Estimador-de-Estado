import numpy as np

def Cov_m(med):
    r_matrix = np.zeros((np.shape(med)[0],np.shape(med)[0]))
    count = 0

    for k in range(np.shape(med)[0]):
#        value = np.zeros(med.shape[1])
        r_matrix[count][count] = med[k][6]
        #r_matrix = np.append(r_matrix, value)
        count = count + 1


    return np.matrix(r_matrix)
    #return r_matrix