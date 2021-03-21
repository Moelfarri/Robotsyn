import numpy as np

# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.
def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.
    p = p0.copy()
    two_n = residualsfun(p).shape[1]
    d = p.shape[0]
    J = np.zeros((two_n,d))
    epsilon_matrix = finite_difference_epsilon*np.eye(d)
    
    
    for iteration in range(num_iterations):
        
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        for j in range(d):
            J[:,j] = (residualsfun(p+epsilon_matrix[:,j]) - residualsfun(p-epsilon_matrix[:,j]))/(2*finite_difference_epsilon)
        
        
        # 2: Form the normal equation terms JTJ and JTr.
        r = residualsfun(p)
        JTJ = J.T@J
        JTr = J.T@r[0] #going from 1x14[[...]] to [...] with 14 elements
        
        delta = np.linalg.inv(JTJ)@(-JTr)
        # 3: Solve for the step delta and update p as
        #    p + step_size*delta
        p = p + step_size*delta
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, num_iterations=100,termination_threshold=1e-3,step_size=1,finite_difference_epsilon=1e-5):
    p = p0.copy()
    ltwo_n = residualsfun(p).shape[1]
    d = p.shape[0]
    J = np.zeros((ltwo_n,d))
    myu = 0
    prev_p = 0
    epsilon_matrix = finite_difference_epsilon*np.eye(d)
    
    for i in range(num_iterations):
        for j in range(d):
            J[:,j] = (residualsfun(p+epsilon_matrix[:,j]) - residualsfun(p-epsilon_matrix[:,j]))/(2*finite_difference_epsilon)
        
        
        r = residualsfun(p)
        JTJ = J.T@J 
        JTr = J.T@r[0]
        
        #initiate myu
        if i == 0:
            myu = 1e-3*np.max(np.diag(JTJ))
        
        delta = np.linalg.inv(JTJ + myu*np.eye(d))@(-JTr)
        
       
        
        #evaluating if step should be accepted or not
        if residualsfun(p+delta)[0].T@residualsfun(p+delta)[0] < residualsfun(p)[0].T@residualsfun(p)[0]:
            prev_p = p
            p = p + delta*step_size
            myu = myu/3
        else:
            myu = 2*myu
            
            
        #print("Iteration: ",i+1, "Error of Angles: ",np.linalg.norm(p-prev_p), " Myu: ",myu, " Delta: ", np.linalg.norm(delta))
        
        
        #Termination condition
        if np.linalg.norm(delta) < termination_threshold:
            return p         
    return p 




