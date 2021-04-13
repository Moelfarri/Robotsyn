function r = lsqnonlin_func(K,R0,P,X,uv,weights)
    R = return_R(R0, P);
    t = P(4:6);
    uv_hat_tilde = K*(R*X + t);
    uv_hat = [uv_hat_tilde(1,:)./uv_hat_tilde(3,:); uv_hat_tilde(2,:)./uv_hat_tilde(3,:)];
    
    n = size(X,2);
    r = zeros(n*2, 1);
    for i = 1:n
       r((i - 1)*2 + 1:i*2) = uv(:,i) - uv_hat(:,i); 
    end
    
    if weights == true
        sigma_u_squared = 50^2;
        sigma_v_squared = 0.1^2;
        Covariance = diag(kron(ones(1,n),[sigma_u_squared, sigma_v_squared]));
        L = chol(Covariance);
        r = L\r;
    end
end

