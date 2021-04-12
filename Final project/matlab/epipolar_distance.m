function e = epipolar_distance(F, uv1, uv2)
    % F should be the fundamental matrix (use F_from_E)
    % uv1, uv2 should be 3 x n homogeneous pixel coordinates

    n = size(uv1, 2);
    e = zeros(1, n); % Placeholder, replace with your implementation
    
    for i=1:n
       u2Fu1 = uv2(:,i).'*F*uv1(:,i);
       abc_uv1 = F*uv1(:,i);
       abc_uv2 = F.'*uv2(:,i);
       e_1 =  u2Fu1/(sqrt(abc_uv1(1)^2 + abc_uv1(2)^2));
       e_2 = u2Fu1/(sqrt(abc_uv2(1)^2 + abc_uv2(2)^2));
       e(i) = (e_1 + e_2)/2;
    end
    
    
end
