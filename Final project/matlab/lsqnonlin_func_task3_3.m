function r = lsqnonlin_func_task3_3(K,R0,P,X,uv)
    yaw = P(1);
    pitch = P(2);
    roll = P(3);
    Rz = [cos(yaw), -sin(yaw), 0;
        sin(yaw), cos(yaw), 0;
        0, 0, 1];

    Ry = [cos(pitch), 0, sin(pitch);
        0, 1, 0;
        -sin(pitch), 0, cos(pitch)];


    Rx = [1, 0, 0;
        0, cos(roll), -sin(roll);
        0, sin(roll), cos(roll)];


    R = Rx*Ry*Rz*R0;
    t = [P(4) P(5) P(6)]';
    T = [R t; zeros(1,3) 1];
     

 
    uv_hat_tilde = K*(T(1:3,1:3)*X + T(1:3,4));
    uv_hat = [uv_hat_tilde(1,:)./uv_hat_tilde(3,:); uv_hat_tilde(2,:)./uv_hat_tilde(3,:)];
    

    n = size(X,2);
    %weights 
    sigma_u = 50;
    sigma_v = 0.1;
    
    %getting the sqrt of measurement covariance matrix as lsqnonlin wants
    %user defined function instead of the sum of squares.
    cov_matrix = eye(2*n);
    cov_matrix(1:n,1:n)     = eye(n)*sigma_u; 
    cov_matrix(n+1:2*n,n+1:2*n) = eye(n)*sigma_v;
   
    
    inv_cov_matrix = inv(cov_matrix);
        
    e_u = (uv_hat(1,:) - uv(1,:));
    e_v = (uv_hat(2,:) - uv(2,:));
    r = [e_u e_v]';
    
    
    %weighted residual function:
    r = inv_cov_matrix*r;
    r = double(r);

    
end