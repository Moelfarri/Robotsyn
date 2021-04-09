import matplotlib.pyplot as plt
import numpy as np
from util import *
import cv2 as cv

# This script uses example data. You will have to modify the
# loading code below to suit how you structure your data.

#MY CODE:
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
matches = bf.knnMatch(des1,des2,k=2)

#Pixel coordinates/point correspondences extracted:
uv1 = np.array([kp1[mat[0].queryIdx].pt for mat in matches]).T
uv2 = np.array([kp2[mat[0].trainIdx].pt for mat in matches]).T


from triangulate_many import *

uv1_tilde = np.vstack((uv1, np.ones(uv1.shape[1])))
uv2_tilde = np.vstack((uv1, np.ones(uv1.shape[1])))

 
xy1 = np.linalg.inv(K)@uv1_tilde
xy2 = np.linalg.inv(K)@uv2_tilde


##MODEL Creation:
retval,rvec,tvec,inliers = cv.solvePnPRansac(xy1.T,xy2[0:2,:].T,K,None,iterationsCount=1000)
R,_ = cv.Rodrigues(rvec)

T_m2q = np.vstack((np.hstack((R,tvec)),np.array([0,0,0,1]))) 
u = uv1


P1 = np.hstack([np.eye(3),np.zeros((3,1))])
P2 = T_m2q[0:3,:]
X = triangulate_many(xy1, xy2, P1, P2)

I = I1 #Query


#TODO: 
#CREATE def localize()


"""
model = '../visualization_sample_data'
query = '../visualization_sample_data/query/IMG_8210'
K       = np.loadtxt(f'{model}/K.txt')       # Intrinsic matrix.
X       = np.loadtxt(f'{model}/X.txt')       # 3D points [shape: 4 x num_points].
T_m2q   = np.loadtxt(f'{query}_T_m2q.txt')   # Model-to-query transformation (produced by your localization script).
matches = np.loadtxt(f'{query}_matches.txt') # Initial 2D-3D matches (see usage code below).
inliers = np.loadtxt(f'{query}_inliers.txt') # Indices of inlier matches (see usage code below).
u       = np.loadtxt(f'{query}_u.txt')       # Image location of features detected in query image (produced by your localization script).
I       = plt.imread(f'{query}.jpg')         # Query image.
"""


assert X.shape[0] == 4
assert u.shape[0] == 2

# If you have colors for your point cloud model, then you can use this.
#c = np.loadtxt('../visualization_sample_data/c.txt') # RGB colors [shape: num_points x 3].
# Otherwise you can use this, which colors the points according to their Y.
c = None


# These control the location and the viewing target
# of the virtual figure camera, in the two views.
# You will probably need to change these to work
# with your scene.
lookfrom1 = np.array((0,-40,2))
lookat1   = np.array((0,0,1))
lookfrom2 = np.array((25,-5,10))
lookat2   = np.array((0,0,10))

# 'matches' is assumed to be a Nx2 array, where the
# first column is the index of the 2D point in the
# query image and the second column is the index of

matches = np.array(matches)

# its matched 3D point.
assert matches.shape[1] == 2

#ISSUE PROBABLY HERE:

mat_u = [mat[0].queryIdx for mat in matches]
mat_x = [mat[1].queryIdx for mat in matches]

u_matches = u[:,mat_u]
X_matches = X[:,mat_x]


# 'inliers' is assumed to be a 1D array of indices
# of the good matches, e.g. as identified by your
# PnP+RANSAC strategy.
inliers = np.reshape(inliers, (inliers.shape[0]))
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
