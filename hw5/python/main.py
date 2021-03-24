import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *
from estimate_E_ransac import * 


K = np.loadtxt('../data/K.txt')
I1 = plt.imread('../data/image1.jpg')/255.0
I2 = plt.imread('../data/image2.jpg')/255.0
matches = np.loadtxt('../data/matches.txt')
# matches = np.loadtxt('../data/task4matches.txt') # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])


#########Task 2#######
xy1_tilde = np.linalg.inv(K)@uv1
xy2_tilde = np.linalg.inv(K)@uv2

xy1 = xy1_tilde[:2,:]/xy1_tilde[2,:]
xy2 = xy2_tilde[:2,:]/xy2_tilde[2,:]

# Task 2: Estimate E
E = estimate_E(xy1, xy2)

#np.random.seed(123) # Leave as commented out to get a random selection each time
#draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
#plt.show()
#########Task 2#######

#########Task 3#######
# Task 3: Triangulate 3D points
P1 = np.hstack([np.eye(3),np.zeros((3,1))])
P2 = decompose_E(E)[0][:3,:] #trying 1 of the 4 matrices

#Better way of choosing P2 based on total positive z values of all X_camera2 correspondences
def choose_best_E_for_P2(decomposed_E):
    max_positive_z  = 0
    
    for E_matrix in decomposed_E:
        P2 = E_matrix[:3,:]
        X = triangulate_many(xy1, xy2, P1, P2)
        X_camera2 = P2@X
        
        possible_max = np.sum((X_camera2[2] >= 0))
        if max_positive_z < possible_max:
            max_positive_z = possible_max
            best_P2_matrix = E_matrix
    return best_P2_matrix

P2 = choose_best_E_for_P2(decompose_E(E))
X = triangulate_many(xy1, xy2, P1, P2)
#draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
#plt.show()
#########Task 3#######


#####Task 4######
#np.random.seed(123) #same random seed!
matches = np.loadtxt('../data/task4matches.txt') # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

xy1_tilde = np.linalg.inv(K)@uv1
xy2_tilde = np.linalg.inv(K)@uv2

xy1 = xy1_tilde[:2,:]/xy1_tilde[2,:]
xy2 = xy2_tilde[:2,:]/xy2_tilde[2,:]


E = estimate_E(xy1, xy2)
F = F_from_E(E, K)
e1,e2 = epipolar_distance(F, uv1, uv2)


#plt.figure(figsize=(10,4))
#_ = plt.hist((e1+e2)/2, bins='auto')
#plt.title("Histogram of residuals - outlier containing correspondences")
#plt.show()


#4.2
#E,xy1,xy2=estimate_E_ransac(xy1, xy2, K, distance_threshold=4, iterations=20000)

#4.3
#E,xy1,xy2=estimate_E_ransac(xy1, xy2, K, distance_threshold=1, iterations=500)

#4.4 - find amount of iterations with formula from Szeliski  (eq 8.30)
E,xy1,xy2=estimate_E_ransac(xy1, xy2, K, distance_threshold=4, iterations=1177)

uv1 = K@np.vstack([xy1,np.ones((1,xy1.shape[1]))])


P1 = np.hstack([np.eye(3),np.zeros((3,1))])
P2 = choose_best_E_for_P2(decompose_E(E)) #trying 1 of the 4 matrices
X = triangulate_many(xy1, xy2, P1, P2)
draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
plt.show()

#####Task 4######