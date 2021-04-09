function p = levenberg_marquardt(residualsfun, p0, precision)
     
    epsilon = 1e-5;
    Epsilon = diag(epsilon*ones(size(p0,1),1));
    N_max = 100;
    j = 0;
    p = p0;
    J = zeros(size(residualsfun(p),1),size(p,1));
    
    %Fint JTJ to estimate the  intial value of mu
    for i = 1:size(p,1)    
        J_temp_right = residualsfun(p + Epsilon(:,i));
        J_temp_left = residualsfun(p - Epsilon(:,i));
        J_temp = (J_temp_right - J_temp_left)/(2*epsilon);
        J(:,i) =J_temp;
    end    
    JTJ = (J.')*J;
    mu = 1e-3 * max(diag(JTJ));
    
    while j < N_max 
        j = j + 1;
        
        %J = zeros(14,3);
        for i = 1:size(p,1)    
            J_temp_right = residualsfun(p + Epsilon(:,i));
            J_temp_left = residualsfun(p - Epsilon(:,i));
            J_temp = (J_temp_right - J_temp_left)/(2*epsilon);
            J(:,i) =J_temp;
        end
        
        JTJ = (J.')*J;
        JTr = (J.')*residualsfun(p);
        delta = -1*(JTJ + mu*eye(size(p,1)))^(-1)*JTr;
%         JTJ_inv = JTJ_inverted(JTJ + mu*eye(size(p,1)));
%         delta = -1*JTJ_inv*JTr;
        
        if (residualsfun(p + delta).')*residualsfun(p + delta) < (residualsfun(p).')*residualsfun(p)
            p = p + delta;
            mu = mu/3;
        else
            mu = 2*mu;
        end 
        
%         norm(delta);
        if norm(delta) < precision
            break
        end
        
    end
end