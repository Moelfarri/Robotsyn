import matplotlib.pyplot as plt
import numpy as np
from util import *
from triangulate_many import *
from estimate_E_ransac import *
from decompose_E import *
import cv2 as cv

#TODO: 
#ESTIMATE CAMERA POSE BY 
#1.PnP solver inside RANSAC
#2.Nonlinear refinement that minimizes reprojection error


K  = np.loadtxt('../visualization_sample_data/K.txt')       # Intrinsic matrix.
I1 = cv.imread("../visualization_sample_data/query/IMG_8207.jpg", cv.IMREAD_GRAYSCALE) #quaryImage
I2 = cv.imread("../visualization_sample_data/query/IMG_8213.jpg", cv.IMREAD_GRAYSCALE) #trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

#SIFT ONLY TAKES 8-bit images - so converting here:
I1 = cv.normalize(I1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
I2 = cv.normalize(I2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')


# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(I1,None)
kp2, des2 = sift.detectAndCompute(I2,None)

# BFMatching of the descriptors/features
bf = cv.BFMatcher()
matches = bf.match(des1,des2)


#Pixel coordinates/point correspondences extracted:
uv1 = np.array([kp1[mat.queryIdx].pt for mat in matches]).T
uv2 = np.array([kp2[mat.trainIdx].pt for mat in matches]).T


uv1_tilde = np.vstack((uv1, np.ones(uv1.shape[1])))
uv2_tilde = np.vstack((uv1, np.ones(uv1.shape[1])))

 
xy1 = np.linalg.inv(K)@uv1_tilde
xy2 = np.linalg.inv(K)@uv2_tilde


#Ransac without PnP: - WORKS:
num_trials = get_num_ransac_trials(8, confidence=0.99, inlier_fraction=0.5)
E,inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold=4, num_trials=num_trials)
#E1 = E

#Ransac with PnPsolver:  - Assuming no lens distortion hence distCoeffs is np.zeros
#SolvePNPRansac - Uses also LevenBerg-Marquardt to minimize the reprojection error
#retval,rvec,tvec,inliers = cv.solvePnPRansac(xy1.T,xy2[0:2,:].T,K,np.zeros((4,1)),iterationsCount=1000) 
#inliers = np.reshape(inliers,(inliers.shape[0]))

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

#R,_ = cv.Rodrigues(rvec)
#t_x = skew(tvec)
#E = t_x@R
#print(E, "E FROM PNPRANSAC")
#print(E1, "E FROM RANSAC HW5")



uv1 = uv1[:,inliers]
uv2 = uv2[:,inliers]
xy1 = xy1[:,inliers]
xy2 = xy2[:,inliers]

T4 = decompose_E(E)
best_num_visible = 0
for i, T in enumerate(T4):
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = T[:3,:]
    X1 = triangulate_many(xy1, xy2, P1, P2)
    X2 = T@X1
    num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
    if num_visible > best_num_visible:
        best_num_visible = num_visible
        best_T = T
        best_X1 = X1
T = best_T
X = best_X1


c = None
T_m2q = T

# These control the location and the viewing target
# of the virtual figure camera, in the two views.
# You will probably need to change these to work
# with your scene.
lookfrom1 = np.array((0,-20,-30))
lookat1   = np.array((0,0,40)) 
lookfrom2 = np.array((45,-5,-30))
lookat2   = np.array((0,0,20))


# 'matches' is assumed to be a Nx2 array, where the
# first column is the index of the 2D point in the
# query image and the second column is the index of


########EVERYTHING FROM HERE ON I AM UNSURE ABOUT:######

# its matched 3D point.
#assert matches.shape[1] == 2


mat_u = [mat.queryIdx for mat in matches]
mat_x = [mat.queryIdx for mat in matches]


#Dont know what "u" should be?
u = uv1
I = I1

u_matches = u[:,mat_u]
X_matches = X[:,mat_x]


# 'inliers' is assumed to be a 1D array of indices
# of the good matches, e.g. as identified by your
# PnP+RANSAC strategy.
u_inliers = u_matches[:,inliers]
X_inliers = X_matches[:,inliers]

u_hat = project(K, T_m2q@X_inliers)
e = np.linalg.norm(u_hat - u_inliers, axis=0)

fig = plt.figure(figsize=(10,8))

plt.subplot(221)
plt.imshow(I)
plt.scatter(*u_hat, marker='+', c=e)
plt.xlim([0, I.shape[1]])
plt.ylim([I.shape[0], 0])
plt.colorbar(label='Reprojection error (pixels)')
plt.title('Query image and reprojected points')

plt.subplot(222)
plt.hist(e, bins=50)
plt.xlabel('Reprojection error (pixels)')

plt.subplot(223)
draw_model_and_query_pose(X, T_m2q, K, lookat1, lookfrom1, c=c)
plt.title('Model and localized pose (top view)')

plt.subplot(224)
draw_model_and_query_pose(X, T_m2q, K, lookat2, lookfrom2, c=c)
plt.title('Model and localized pose (side view)')

plt.tight_layout()
plt.show()