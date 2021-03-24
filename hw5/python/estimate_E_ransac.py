import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *
from tqdm import tqdm

def estimate_E_ransac(xy1, xy2, K, distance_threshold, iterations):

    m = 8
    max_inlier_count = 0
    
    for i in tqdm(range(iterations)):
        sample = np.random.choice(xy1.shape[1], size=m, replace=False)
        E = estimate_E(xy1[:,sample], xy2[:,sample])
        F = F_from_E(E, K)
        
        uv1 = K@np.vstack([xy1,np.ones((1,xy1.shape[1]))])
        uv2 = K@np.vstack([xy2,np.ones((1,xy1.shape[1]))])
        e1,e2 = epipolar_distance(F, uv1, uv2) 
        r = (e1+e2)/2 #residuals
        
        #count the number of residuals within a specified threshold (inlier count)
        inlier_counter = 0
        for residual,i in zip(r,range(r.shape[0])):
            if np.abs(residual) <  distance_threshold:
                inlier_counter += 1
            
        #get best E and all the corresponding inliers
        if max_inlier_count < inlier_counter:
            max_inlier_count = inlier_counter
            best_E = E
            inliers = np.where(np.abs(r) < distance_threshold)
        

    
    inliers = np.array(inliers)
    inliers = inliers[0,:]
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]
    
    print("Inliers found:", max_inlier_count)
    
                
    #return the essential matrix and associated inlier set that had highest inlier count
    return best_E, xy1,xy2
    
    

     
