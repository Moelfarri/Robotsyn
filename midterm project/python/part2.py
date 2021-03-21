import matplotlib.pyplot as plt
import numpy as np
from common import *
from methods import *


K           = np.loadtxt('../data/K.txt')
detections  = np.loadtxt('../data/detections.txt')
XY01        = np.loadtxt('../data/platform_corners_metric.txt') #metric
uv          = np.loadtxt('../data/platform_corners_image.txt') #pixel coordinates

XY          = np.loadtxt('../data/platform_corners_metric.txt')[:2,:]

n_uv = uv.shape[1]
n_XY = XY.shape[1]
XY1 = np.vstack((XY, np.ones(n_XY)))




uv_tilde = np.vstack((uv, np.ones(n_uv))) #add layer of 1 under
xy_tilde = np.linalg.inv(K)@uv_tilde
xy = xy_tilde/xy_tilde[2,:]

#Task 2.1a:
H = estimate_H(xy,XY)
uv_hat_a  = project(K, H@XY1)

#Task 2.1b:
T1,T2 = decompose_H(H)
T = choose_correct_pose(T1,T2,XY01) 
T[:3,:3] = closest_rotation_matrix(T[:3,:3])

uv_hat_b  = project(K, T@XY01)
#Up to this point is correct


#Task 2.2:
p = np.zeros((6))
t0 = T[:3,3]
R0 = T[:3,:3]
p[3:6] = t0

class Platform:
    def __init__(self):
        self.T = np.zeros((4,4))
        self.R = np.zeros((4,4))
        self.uv_hat = 0
        
    def rotate_x(self, radians):
        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[1, 0, 0],
                         [0, c,-s],
                         [0, s, c]])

    def rotate_y(self, radians):
        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    def rotate_z(self, radians):
        c = np.cos(radians)
        s = np.sin(radians)
        return np.array([[c,-s, 0],
                         [s, c, 0],
                         [0, 0, 1,]])
        
    def residuals(self,uv,K,R0,XY01, p0):
        T = np.zeros((4,4))
        p = p0.copy()  
        R = self.rotate_x(p[0])@self.rotate_y(p[1])@self.rotate_z(p[2])@R0
        t = [p[3],p[4],p[5],1]
        T[:3,:3] = R
        T[:,3]   = t
        
 
        
        
        uv_hat   = project(K,T@XY01)
        self.T   = T
        self.uv_hat = uv_hat
        self.R   = R
                         
        n = uv_hat.shape[1]
        r = np.zeros((1,2*n))
        
        r[:,0:n]   = (uv_hat[0,:]-uv[0,:])
        r[:,n:2*n] = (uv_hat[1,:]-uv[1,:])
        
        
        
        return r
        
    
    
platform = Platform()
residualsfun = lambda p : platform.residuals(uv, K, R0, XY01,p)
p = levenberg_marquardt(residualsfun, p,termination_threshold=1e-6)
uv_hat_lm = platform.uv_hat
T_lm = platform.T
#print(platform.R.T@platform.R) #LBA takes DLT projection and makes also the rotation work
#print("error LM:", reprojection_error(uv, platform.uv_hat).T)


###############
#Task 2.3: - must be redone..
p23 = np.zeros((6))    
R0 = platform.T[:3,:3]
p23[3:6] = T_lm[:3,3] #translation from DLT
residualsfun1 = lambda p23 : platform.residuals(uv[:,0:3], K, R0, XY01[:,0:3],p23)
p23 = levenberg_marquardt(residualsfun1, p23,termination_threshold=1e-6) #threshold makes uvhat23 = uvhatlm


uv_hat_23 = platform.uv_hat
print("POINTS:")
print(uv_hat_lm)
print(uv_hat_23)
print("TRANSFORMS:")
print("2.2: ", T_lm)
print("2.3: ", platform.T)
print("SOLUTIONS (P):")
print("2.2: ", p)
print("2.3: ", p23)
print("REPROJECTION ERROR:")
print("2.2: ", reprojection_error(uv, uv_hat_lm))
print("2.3: ", reprojection_error(uv[:,0:3], uv_hat_23))



###########
image_number = 0
#Figure plot of heli platform:
fig = plt.figure(figsize=plt.figaspect(0.35))
fig.suptitle('Image number %d' % image_number)
I = plt.imread('../data/video%04d.jpg' % image_number)
plt.plot()
plt.imshow(I)
plt.scatter(*uv, linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
plt.scatter(uv_hat_a[0,:], uv_hat_a[1,:], marker='o', color='red', label='Predicted A',s=20)
plt.scatter(uv_hat_b[0,:], uv_hat_b[1,:], marker='v', color='cyan', label='Predicted B',s=10)
#plt.scatter(uv_hat_lm[0,:], uv_hat_lm[1,:], marker='o', color='green', label='Task 2.2',s=20)
#plt.scatter(uv_hat_23[0,:], uv_hat_23[1,:], marker='o', color='blue', label='Task 2.3',s=20)
plt.legend()
#close up
plt.xlim([100, 600])
plt.ylim([600, 300])
plt.savefig('task2.1.png')
plt.show()
