import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *
from tqdm import tqdm

def estimate_E_ransac(xy1, xy2, K, distance_threshold, iterations):

    m = 8
    inlier_count_array = []
    sample_array = []
    
    
    
    for i in tqdm(range(iterations)):
        sample = np.random.choice(xy1.shape[1], size=m, replace=False)
        sample_array.append(sample)
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
               
        
        inlier_count_array.append(inlier_counter)
                
    
    max_inlier_count = max(inlier_count_array)
    max_inlier_index = inlier_count_array.index(max(inlier_count_array))
    best_sample = sample_array[max_inlier_index]
    E = estimate_E(xy1[:,best_sample], xy2[:,best_sample])
    print("Size of inlier set:", xy1[:,best_sample].shape)
    
    #return the essential matrix and associated inlier set that had highest inlier count
    return E,xy1[:,best_sample],xy2[:,best_sample]
    
    

     
