import numpy as np

def make_blocks(X,t):
    xdim = range(0,X.shape[0],t)
    ydim = range(0,X.shape[1],t)
    zdim = range(0,X.shape[2],t)
    
    reshaped = np.zeros((X.shape[0]//t*X.shape[1]//t*X.shape[2]//t,t,t,t))
    block = 0
    for x in xdim:
        for y in ydim:
            for z in zdim:
                reshaped[block] = X[x:x+t,y:y+t,z:z+t]
                block += 1
    return reshaped
