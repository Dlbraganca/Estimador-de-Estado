
def get_v(x, busFrom, busTo, nBus):
    ang = nBus - 1
    v1 = x[ang + busFrom - 1].item(0)
    v2 = x[ang + busTo - 1].item(0)
    return v1, v2