function r = lsqnonlin_func(K,R0,P,X,uv)
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
    r = zeros(n*2, 1);
     for i = 1:n
        r((i - 1)*2 + 1:i*2) = uv(:,i) - uv_hat(:,i); 
     end

end