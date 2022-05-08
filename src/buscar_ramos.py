   
def find_branch(sys, busFrom, busTo):
    for i in sys:
        if (i[0] == busFrom and i[1] == busTo) or (i[0] == busTo and i[1] == busFrom):
            g = (1/(i[2]+i[3])).real
            b = (1/(i[2]+i[3])).imag
            gs = (i[4]+i[5]).real
            bs = (i[4]+i[5]).imag
            return g, b, gs, bs
    return None, None, None, None