import numpy as np
import matplotlib.pyplot as plt
from common import *

#load in data
K     = np.loadtxt('../data/heli_K.txt')
T_P2C = np.loadtxt('../data/platform_to_camera.txt')
#load picture
img = plt.imread('../data/quanser.jpg')
height,width,_ = img.shape

#########Task 4.1 Defining 4 vectors representing screws in platform: (Note that its defined in meters and not cm so 11.45 cm => 0.1145m)
X_P = np.array([[0,  0.1145, 0,      0.1145],  #<-- coordinates given in platform frame
                 [0, 0,      0.1145, 0.1145],
                 [0, 0,      0,      0],
                 [1, 1,      1,      1]])

X_C = T_P2C @ X_P   #<-- coordinates transformed to camera coordinates

K    = K_2_4x4(K) #dimension transformation

#projection
u,v = project(K, X_C)
##########################################

#Plotting the helicopter platform for task 4.2
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='white', marker='.', s=60)
plt.axis('image')
plt.imshow(img)
plt.xlim([100, 600])
plt.ylim([600, 300]) 
draw_frame(K, T_P2C, scale=0.12)
#plt.savefig("task4.2_frame.png")
plt.show()

#task 4.3
T_B2P = translate_x(0.1145/2)@translate_y(0.1145/2)@rotate_z(np.deg2rad(11.6)) 
T_B2C = T_P2C@T_B2P


#task 4.4
T_H2B = translate_z(0.325)@rotate_y(np.deg2rad(28.9)) 
T_H2C = T_P2C@T_B2P@T_H2B

#task 4.5
T_A2H = translate_z(-0.05) 
T_A2C = T_P2C@T_B2P@T_H2B@T_A2H

#task 4.6
T_R2A = translate_x(0.65)@translate_z(-0.03)@rotate_x(np.deg2rad(0)) 
T_R2C = T_P2C@T_B2P@T_H2B@T_A2H@T_R2A

#4.7
X   = np.loadtxt('../data/heli_points.txt').T #shape is (4,7) when transposed first 3 vectors in arm frame and last 4 are rotor frame
X_A = X[:,0:3]
X_R = X[:,3::]

X_A2C = T_A2C@X_A
X_R2C = T_R2C@X_R

u1, v1 = project(K, X_A2C)
u2, v2 = project(K, X_R2C)

#Plotting task 4.3 to 4.7
plt.figure(figsize=(12,7))
plt.scatter(u1, v1, c='orange', marker='.', s=500)
plt.scatter(u2, v2, c='orange', marker='.', s=500)
plt.axis('image')
plt.imshow(img)
plt.xlim([0, width])
plt.ylim([height, 0]) 
plt.suptitle('Helicopter Coordinate frames', fontsize=12)
draw_frame(K, T_P2C, scale=0.05) #platform frame
draw_frame(K, T_B2C, scale=0.05) #base     frame
draw_frame(K, T_H2C, scale=0.05) #hinge    frame
draw_frame(K, T_A2C, scale=0.03) #arm      frame
draw_frame(K, T_R2C, scale=0.05) #Rotor    frame
#plt.savefig("task4.7_helicopter_frames.png")
plt.show()
