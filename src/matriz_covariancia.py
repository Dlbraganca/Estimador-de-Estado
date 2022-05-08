import numpy as np

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

    return np.matrix(r_matrix)