function [P,X] = LM(N_max, precision, R_cell, K , P, X, uv_cell)
%we want to optimize over P_cel and X_cell

num_images = size(uv_cell,1);
num_correpondences = size(uv_cell{1},2);
j = 0;

[J, ~] = return_J_and_JTr(num_images, num_correpondences, R_cell, K , P, X, uv_cell);
JTJ = J.'*J;
mu = 1e-3* max(diag(JTJ));

while j < N_max
    j = j + 1
        
     [J, JTr] = return_J_and_JTr(num_images, num_correpondences, R_cell, K , P, X, uv_cell);
     JTJ = (J.')*J;
     
     %must add mu as diagonal value
     a = mu*eye(size(JTJ,1));
     JTJ_inv = JtJ_inverted(JTJ + a,num_images);
     delta = -1*JTJ_inv*JTr;
    
     r = residual(R_cell,P,X,uv_cell,K,2);
     [P_delta,X_delta] = add_delta(P,X,delta);
     r_delta = residual(R_cell,P_delta,X_delta,uv_cell,K,2);

     if r_delta.' *r_delta < r.' *r
         X = X_delta;
         P = P_delta;
         mu = mu/3;
     else
         mu = 5*mu;
     end
     
     
     if norm(delta) < precision
         break
     end
     
end
end

