import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    e1 = np.zeros(n) 
    e2 = np.zeros(n)
    
    for i in range(n):
        e1[i] = uv2[:,i].T@F@uv1[:,i]/np.sqrt((F.T@uv2[:,i])[0]**2 + (F.T@uv2[:,i])[1]**2)
        e2[i] = uv1[:,i].T@F.T@uv2[:,i]//np.sqrt((F@uv1[:,i])[0]**2 + (F@uv1[:,i])[1]**2)
    return e1,e2
