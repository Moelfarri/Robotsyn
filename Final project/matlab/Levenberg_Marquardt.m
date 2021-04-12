function [P,X] = Levenberg_Marquardt(N_max, precision, R_cell, K , P, X, uv_cell)
%we want to optimize over P_cel and X_cell

num_images = size(uv_cell,1);
num_correpondences = size(uv_cell{1},2);
j = 0;

[J, ~] = return_J_and_JTr(num_images, num_correpondences, R_cell, K , P, X, uv_cell);
[U, V, W] = JtJ(num_images,J);
JTJ = [U, W;
        W.', V];
mu = 1e-3* max(diag(JTJ));

while j < N_max
    j = j + 1
        
     [J, JTr] = return_J_and_JTr(num_images, num_correpondences, R_cell, K , P, X, uv_cell);
     [U, V, W] = JtJ(num_images,J);
     
     U = U + mu*eye(size(U,1));
     V = V + mu*eye(size(V,1));
     V_inv = A22_inverted(V);
     epsilon_a = JTr(1:6*num_images);
     epsilon_b = JTr(6*num_images + 1:end);
     
     delta_a = (U - W*V_inv*W.')^(-1)*(epsilon_a - W*V_inv*epsilon_b);
     delta_b = V_inv*(epsilon_b - W.'*delta_a);
     delta = -[delta_a',delta_b']';

    
     %above here i need to make delta
     r = residual(R_cell,P,X,uv_cell,K,2);
     [P_delta,X_delta] = add_delta(P,X,delta);
     r_delta = residual(R_cell,P_delta,X_delta,uv_cell,K,2);
     if r_delta.' *r_delta < Huber_norm(r.' *r)
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

