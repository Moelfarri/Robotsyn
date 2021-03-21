import numpy as np

def schur_compliment_inverse(A_11,A_12,A_22):
    m  = A_11.shape[0]
    l3 = A_12.T.shape[0]
    n = m+l3
    A = np.zeros((n,n))
   
    inv_A_22 = optimize_A_22_inverse(A_22)    
    
    M_D      = A_11 - A_12@inv_A_22@A_12.T
    inv_M_D  = np.linalg.inv(M_D)
    
    A[0:m,0:m]   =  inv_M_D                  #A11 area
    A[0:m,m:m+l3]  = -inv_M_D@A_12@inv_A_22    #A12 area 
    A[m:m+l3,0:m]  = -inv_A_22@A_12.T@inv_M_D  #A12.T area
    A[m:m+l3,m:m+l3] =  inv_A_22 + inv_A_22@A_12.T@inv_M_D@A_12@inv_A_22 #A22 area 
    return A

def optimize_A_22_inverse(A_22):
    images = int(A_22.shape[0]/3)
    for i in range(images):
        A_22[3*i:3*(i+1),3*i:3*(i+1)] = np.linalg.inv(A_22[3*i:3*(i+1),3*i:3*(i+1)])
    return A_22
    

#The residualsfun is the most expensive call in this function
#this code should run for around 1 minute and 30 seconds 
#The matlab implementation was totally overhauled 
#and further optimized and took only 8 seconds in runtime
def optimized_levenberg_marquardt(residualsfun, p0, num_iterations=100,termination_threshold=1e-3,step_size=1,finite_difference_epsilon=1e-5):
    p = p0.copy()
    ltwo_n = residualsfun(p,0,0).shape[1]
    d = p.shape[0]
    J = np.zeros((ltwo_n,d))
    myu = 0
    prev_p = 0
    epsilon_matrix = finite_difference_epsilon*np.eye(d)
    
    
    
    for i in range(num_iterations):
        
        
        counter = 0
        image_number = 0
        for j in range(d):
            #if dense, else do sparse differentiation
            if j < 26:
                J[:,j] = (residualsfun(p+epsilon_matrix[:,j],0,0) - residualsfun(p-epsilon_matrix[:,j],0,0))/(2*finite_difference_epsilon)
            else:
                J[14*image_number:14*image_number+14,j] = (residualsfun(p+epsilon_matrix[:,j],1,image_number)- residualsfun(p-epsilon_matrix[:,j],1,image_number))/(2*finite_difference_epsilon)

                #shift 14 rows down after 3 14x1 columns get differentiated
                counter += 1 
                if counter == 3:
                    counter = 0
                    image_number += 1 #shifting variable
                    
                    
        
        
        r = residualsfun(p,0,0)
        JTJ = J.T@J 
        JTr = J.T@r[0]
        
        #initiate myu
        if i == 0:
            myu = 1e-3*np.max(np.diag(JTJ))
        
        #Schur's compliment for inversion of matrix:
        m  = p[0:26].shape[0]
        l3 = p[26:].shape[0]
        JTJ_damp = JTJ + myu*np.eye(d)
        A_11 = JTJ_damp[0:m,0:m]
        A_12 = JTJ_damp[0:m,m:m+l3]
        A_22 = JTJ_damp[m:m+l3,m:m+l3]
        inv_JTJ_damp = schur_compliment_inverse(A_11,A_12,A_22)
        
        
        
        delta = inv_JTJ_damp@(-JTr)
        
        #evaluating if step should be accepted or not
        rdelta = residualsfun(p+delta,0,0)
        if rdelta[0].T@rdelta[0] < r[0].T@r[0]:
            prev_p = p
            p = p + delta*step_size
            myu = myu/3
        else:
            myu = 2*myu
            
        #print("Iteration: ",i+1, "Error of Angles: ",np.linalg.norm(p-prev_p), " Myu: ",myu, " Delta: ", np.linalg.norm(delta))
        
        print(delta)
        print("iteration:", i)
        #Termination condition
        if np.linalg.norm(delta) < termination_threshold:
            return p         
    return p 