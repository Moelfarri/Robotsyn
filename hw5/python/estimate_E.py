import numpy as np

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.zeros((n, 9))
    
    for i in range(n):
        x1 = xy1[0,i]
        y1 = xy1[1,i]
        x2 = xy2[0,i]
        y2 = xy2[1,i]
        
        A[i,:] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    #get V.T:
    _,_,VT = np.linalg.svd(A)
    V = VT.T
    e = V[:,-1]

    E = np.reshape(e,(3,3))    
    return E
