function R = return_R(R0, P)
    %phi theta psi
    Rz = [cos(P(3)), -sin(P(3)), 0;
        sin(P(3)), cos(P(3)), 0;
        0, 0, 1];
    Ry = [cos(P(2)), 0, sin(P(2));
        0, 1, 0;
        -sin(P(2)), 0, cos(P(2))];
    
    Rx = [1, 0, 0;
        0, cos(P(1)), -sin(P(1));
        0, sin(P(1)), cos(P(1))];
    
    R = Rx*Ry*Rz*R0;
end

