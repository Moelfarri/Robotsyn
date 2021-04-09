function r = residuals(X,K, uv,R0,yaw, pitch, roll,tx, ty, tz)
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
    t = [tx, ty, tz]';
    T = [R, t; zeros(1,3), 1];

    uv_hat = project(K,T*X);

    e_u = (uv_hat(1,:) - uv(1,:));
    e_v = (uv_hat(2,:) - uv(2,:));
    r = [e_u, e_v]';
end


