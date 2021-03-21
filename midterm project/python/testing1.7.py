import matplotlib.pyplot as plt
import numpy as np
from methods import *
from generate_quanser_summary import *
from common import*




class Quanser:
    def __init__(self):
        self.K = np.loadtxt('../data/K.txt')
        self.heli_points = np.loadtxt('../data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
        self.detections = np.loadtxt('../data/detections.txt')

    def residuals(self, uv, weights, angles, p=np.array([0.1145, 0.325,-0.050,0.65,-0.030])):
        # Compute the helicopter coordinate frames
        base_to_platform = translate(p[0]/2, p[0]/2, 0.0)@rotate_z(angles[0])
        hinge_to_base    = translate(0.00, 0.00,  p[1])@rotate_y(angles[1])
        arm_to_hinge     = translate(0.00, 0.00, p[2])
        #rotors_to_arm    = translate(p[3], 0.00, p[4])@rotate_x(angles[2])
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        #self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        uv_hat = project(self.K, p1)
        self.uv_hat = uv_hat # Save for use in draw()
        
        
        self.uv_hat = uv_hat[:,0]
        n =1
        r = np.zeros((1,2*n))

        r[:,0]   = (self.uv_hat[0]-uv[0])
        r[:,1] = (self.uv_hat[1]-uv[1])
        return r
            
      
        

    def draw(self, uv, weights, image_number):
        I = plt.imread('../data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        #draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection.png')

detections = np.loadtxt('../data/detections.txt')

# The script runs up to, but not including, this image.
run_until = 1 # Task 1.3
visualize_number = 0

quanser = Quanser()

# Initialize the parameter vector 

p = np.array([0.0, 0.0]) 


all_residuals = []
trajectory = np.zeros((run_until, 2))
for image_number in range(run_until):
    weights = detections[image_number, ::3]
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    
    
    #only 1 correspondence
    weights = weights[0]
    uv = uv[:,0]
    
    
    
    residualsfun = lambda p : quanser.residuals(uv, weights, p)
    
    p = levenberg_marquardt(residualsfun, p)

    p = np.array([0.1320,0.4662]) 
    p = np.array([0.1320-2*np.pi,0.4662-2*np.pi])
    
    r = residualsfun(p)
    all_residuals.append(r)
    trajectory[image_number,:] = p
    if image_number == visualize_number:
        print('Residuals on image number', image_number, r)
        quanser.draw(uv, weights, image_number)


# Note:
# The generated figures will be saved in your working
# directory under the filenames out_*.png.

#generate_quanser_summary(trajectory, all_residuals, detections)
plt.show()
