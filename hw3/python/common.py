import numpy as np
import matplotlib.pyplot as plt

#
# Tip: Define functions to create the basic 4x4 transformations
#

#translation definitions
def K_2_4x4(K):
    temp = np.zeros((4,4))
    temp[0:3,0:3] = K
    K    = temp
    return K

def translate_x(x):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate_y(y):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, y],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate_z(z):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

#Rotation definitions
def rotate_x(radians):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(radians), -np.sin(radians), 0],
                     [0, np.sin(radians), np.cos(radians), 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    return np.array([[np.cos(radians), 0, np.sin(radians), 0],
                     [0, 1, 0, 0],
                     [-np.sin(radians), 0, np.cos(radians), 0],
                     [0, 0, 0, 1]])


def rotate_z(radians):
    return np.array([[np.cos(radians), -np.sin(radians), 0, 0],
                     [np.sin(radians), np.cos(radians), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


#
# Note that you should use np.array, not np.matrix,
# as the latter can cause some unintuitive behavior.
#
# translate_x/y/z could alternatively be combined into
# a single function.

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """

    # Tip: Use the @ operator for matrix multiplication, the *
    # operator on arrays performs element-wise multiplication!
    U_tilde = K@X

    N = X.shape[1]
    u = np.zeros((1,N))
    v = np.zeros((1,N))
    for i in range(N):
        Z      = U_tilde[2,i]
        u[:,i] = U_tilde[0,i]/Z
        v[:,i] = U_tilde[1,i]/Z
        
        
    uv = np.zeros((2,N))
    uv[0,:] = u
    uv[1,:] = v
    #pretty shitty implementation tbh... but idont care..
    
    return uv

def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    This uses your project function, so implement it first.

    Control the length of the axes using 'scale'.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis
