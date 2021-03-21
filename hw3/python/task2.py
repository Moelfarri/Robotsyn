import numpy as np
import matplotlib.pyplot as plt
from common import *

# Tip: Use np.loadtxt to load data into an array 

#Coordinate for task 2.2: 
K = np.loadtxt('../data/task2K.txt')
#X = np.loadtxt('../data/task2points.txt')
#u,v = project(K, X)

#Coordinates for task 3.2:
X_o    = np.loadtxt('../data/task3points.txt')
K_4    = np.zeros((4,4))
K_4[0:3,0:3] = K
print(X_o.shape)
 
#To get the coordinates from task 2.2
T_o2c  = translate_z(6)@rotate_x(np.deg2rad(15))@rotate_y(np.deg2rad(45)) #rekkefølgen på transformasjonen betyr noe
X_c    = T_o2c@X_o
u,v = project(K_4, X_c)





# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400


# Figure for Task 2.2: Show pinhole projection of 3D points
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!!!!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
#plt.savefig("task2.2_project_of_points.png")
draw_frame(K_4, T_o2c, scale=1)        #frame that is associated with task 3.2 transformation
#plt.savefig("task3.2_project_of_points.png")
plt.show()


 