import matplotlib.pyplot as plt
import numpy as np
from common import *

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
        rotors_to_arm    = translate(p[3], 0.00, p[4])@rotate_x(angles[2])
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()

        n = self.uv_hat.shape[1]
        r = np.zeros((1,2*n))
        r[:,0:n]   = (self.uv_hat[0,:]-uv[0,:])*weights
        r[:,n:2*n] = (self.uv_hat[1,:]-uv[1,:])*weights
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
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection.png')

