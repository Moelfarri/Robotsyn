import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#Load Data to extract essential matrices from different sampling methods
K   = np.loadtxt('../our_own_data_images_and_figures/Task45/K.txt')
uv1 = np.loadtxt('../our_own_data_images_and_figures/Task45/uv1.txt')
uv2 = np.loadtxt('../our_own_data_images_and_figures/Task45/uv2.txt')

uv1 = uv1.reshape(-1,1,2)
uv2 = uv2.reshape(-1,1,2)

#The reason why i use findFundamentalMat instead of findEssentialMat
#is because openCV needs LAPACK or EIGEN libraries for findEssentialMat
#So a work around is finding Fundamental matrix and converting it to
#Essential matrix with intrinsic matrix K

#LMEDS
F_LMEDS,inliers_LMEDS = cv.findFundamentalMat(uv1,uv2,cv.LMEDS)    
E_LMEDS = K.T@F_LMEDS@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_LMEDS.txt', E_LMEDS)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_LMEDS.txt', inliers_LMEDS)

#RHO
F_RHO,inliers_RHO = cv.findFundamentalMat(uv1,uv2,cv.RHO)
E_RHO = K.T@F_RHO@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_RHO.txt', E_RHO)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_RHO.txt', inliers_RHO)

#LO-RANSAC
F_LO,inliers_LO = cv.findFundamentalMat(uv1,uv2,cv.USAC_DEFAULT)
E_LO = K.T@F_LO@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_LO.txt', E_LO)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_LO.txt', inliers_LO)


#LO-RANSAC + RANSAC
F_LOPR,inliers_LOPR = cv.findFundamentalMat(uv1,uv2,cv.USAC_PARALLEL)
E_LOPR = K.T@F_LOPR@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_LOPR.txt', E_LOPR)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_LOPR.txt', inliers_LOPR)


#Prosac
F_PROSAC,inliers_PROSAC = cv.findFundamentalMat(uv1,uv2,cv.USAC_PROSAC)
E_PROSAC = K.T@F_PROSAC@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_PROSAC.txt', E_PROSAC)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_PROSAC.txt', inliers_PROSAC)


#GC-RANSAC
F_GCRANSAC,inliers_GCRANSAC = cv.findFundamentalMat(uv1,uv2,cv.USAC_ACCURATE)
E_GCRANSAC = K.T@F_GCRANSAC@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_GCRANSAC.txt', E_GCRANSAC)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_GCRANSAC.txt', inliers_GCRANSAC)


#MAGSAC++
F_MAGSAC,inliers_MAGSAC = cv.findFundamentalMat(uv1,uv2,cv.USAC_MAGSAC)
E_MAGSAC = K.T@F_MAGSAC@K
np.savetxt('../our_own_data_images_and_figures/Task45/E_MAGSAC.txt', E_MAGSAC)
np.savetxt('../our_own_data_images_and_figures/Task45/inliers_MAGSAC.txt', inliers_MAGSAC)

#the point is to save everything into a txt file and then utilize the script
#for modeling that we constructed in MATLAB for task 2 - this way we eliminate
#the risk for doing double the work, but get the possibility to experiment with
#different sampling methods