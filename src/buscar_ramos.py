import numpy as np   

def find_branch(sys, busFrom, busTo):
    for i in sys:
        if (i[0] == busFrom and i[1] == busTo) or (i[0] == busTo and i[1] == busFrom):
            y = 1/np.complex(i[2].item() + i[3].item()*1j)
            g = (y).real
            b = (y).imag
            gs = i[4].item()/2
            bs = i[5].item()/2
            # #VERSÃO SISTEMA Y
            # g = i[2].item(0)
            # b = i[3].item(0)
            # gs = i[4].item(0)/2
            # bs = i[5].item(0)/2
            
            # #VERSÃO SISTEMA Z
            # y = 1/np.complex(i[2] + i[3]*1j)
            # ys = 1/(np.complex(i[4] + i[5]*1j)/2)
            # g = y.real()
            # b = y.imag()
            # gs = ys.real()
            # bs = ys.imag()
            return g, b, gs, bs
    return 0, 0, 0, 0