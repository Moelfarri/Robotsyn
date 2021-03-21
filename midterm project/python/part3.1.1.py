import matplotlib.pyplot as plt
import numpy as np
from common import *
from methods import * 
from generate_quanser_summary import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('../data/K.txt')
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
        self.detections = np.loadtxt('../data/detections.txt')
        self.uv_hat = 0
        self.weights = 0 
        self.uv     = 0
        
        
    def residuals_part1(self, i, angles, m):
        N = 7
        M = 5 + 3*N
 
        
        yaw = angles[0]
        pitch = angles[1]
        roll  = angles[2]
        weights = detections[i, ::3]
        uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))

        #visualize image0:
        if i == 0:
            self.weights = weights
            self.uv     = uv

        base_to_platform = translate(m[0]/2, m[0]/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  m[1])@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, m[2])
        rotors_to_arm    = translate(m[3], 0.00, m[4])@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm


        m_3D = m[5:M].reshape(3,N,order="F")
        p1 = self.arm_to_camera @np.vstack((m_3D[:,:3], np.ones(m_3D[:,:3].shape[1]))) #m_3D[:,:3]
        p2 = self.rotors_to_camera @np.vstack((m_3D[:,3:], np.ones(m_3D[:,3:].shape[1]))) #m_3D[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))

        #registering it
        self.uv_hat = uv_hat
    

        r = np.zeros((1,2*N))
        r[:,0:N]   = (self.uv_hat[0,:]-uv[0,:])*weights
        r[:,N:2*N] = (self.uv_hat[1,:]-uv[1,:])*weights
        return r   

    def residuals_part3(self, images, angles, m):
        N = 7
        M = 5 + 3*N
        L = images
        r = np.zeros((1,2*N*L))
        e_u = np.zeros((1,N))
        e_v = np.zeros((1,N))
 
        for i in range(L):
            yaw = angles[i,0]
            pitch = angles[i,1]
            roll  = angles[i,2]
            weights = detections[i, ::3]
            uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))

 
            base_to_platform = translate(m[0]/2, m[0]/2, 0.0)@rotate_z(yaw)
            hinge_to_base    = translate(0.00, 0.00,  m[1])@rotate_y(pitch)
            arm_to_hinge     = translate(0.00, 0.00, m[2])
            rotors_to_arm    = translate(m[3], 0.00, m[4])@rotate_x(roll)
            self.base_to_camera   = self.platform_to_camera@base_to_platform
            self.hinge_to_camera  = self.base_to_camera@hinge_to_base
            self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
            self.rotors_to_camera = self.arm_to_camera@rotors_to_arm


            m_3D = m[5:M].reshape(3,N,order="F")
            p1 = self.arm_to_camera @np.vstack((m_3D[:,:3], np.ones(m_3D[:,:3].shape[1]))) #m_3D[:,:3]
            p2 = self.rotors_to_camera @np.vstack((m_3D[:,3:], np.ones(m_3D[:,3:].shape[1]))) #m_3D[:,3:]
            uv_hat = project(self.K, np.hstack([p1, p2]))
            self.uv_hat = uv_hat

            
            #TRYING DIFFERENT RESIDUAL FORMTAING HERE:
            e_u = (self.uv_hat[0,:]-uv[0,:])*weights
            e_v = (self.uv_hat[1,:]-uv[1,:])*weights
            
            r[:,2*N*i:2*N*i+2*N] = np.hstack((e_u,e_v)) 
            #r[:,i:i+14] = np.hstack((e_u,e_v)) #residuals fra sia
        return r
    
    
    

    
    def draw(self):
        I = plt.imread('../data/video%04d.jpg' % 0)
        plt.imshow(I)
        plt.scatter(*self.uv[:, self.weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % 0)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection.png')


    
####ESTIMATE ANGLES (p1) - CONSTANT STATICS (PART 1) - Step 1:
detections = np.loadtxt('../data/detections.txt')
heli_points = np.loadtxt('../data/heli_points.txt').T[:3,:]
L = 351
N = 7    
M = 5 + 3*N

m = np.zeros((M)) #statics - constant this time
m[:5] = [0.1145, 0.325,-0.050,0.65,-0.030] #Lengths 
m[5:M] = heli_points.reshape(21,order="F") #3D points

quanser = Quanser()

p1 = np.array([0.0,0.0,0.0]) #initial angle estimates
trajectory = np.zeros((L,3)) #Register the estimated trajectory
all_residuals = [] #register residuals

for i in range(L):
    residualsfun1 = lambda p1 : quanser.residuals_part1(i, p1, m)
    p1 = levenberg_marquardt(residualsfun1, p1, termination_threshold=1e-6)
    trajectory[i,:] = p1
    #r = residualsfun1(p1)
    #all_residuals.append(r)
    #uncomment to show part1 helicopter
    #if i == 0:
    #    quanser.draw()



####ESTIMATE STATICS (m) - USE ANGLES FROM PART1 - Step 2:
residualsfun2 = lambda m : quanser.residuals_part3(L, trajectory, m)
m = levenberg_marquardt(residualsfun2, m,termination_threshold=1e-6)
r = residualsfun2(m)


 

###USE ESTIMATED STATICS TO FIND ANGLES (p3) - Step 3:
p3 = np.array([0.0,0.0,0.0]) #initial angle estimates
for i in range(L):
    residualsfun3 = lambda p3 : quanser.residuals_part1(i, p3, m)
    p3 = levenberg_marquardt(residualsfun3, p3, termination_threshold=1e-6)
    trajectory[i,:] = p3
    r = residualsfun1(p3)
    all_residuals.append(r)
    if i == 0:
        quanser.draw()


generate_quanser_summary(trajectory, all_residuals, detections)
plt.show()
