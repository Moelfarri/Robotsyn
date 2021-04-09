function T = return_T(R0, yaw, pitch, roll, tx, ty, tz)

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
    T = [R, t;zeros(1,3), 1];

end