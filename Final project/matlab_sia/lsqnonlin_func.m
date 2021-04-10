function r = lsqnonlin_func(K,P,X,uv)
    uv_hat_tilde = K*(P(1:3,1:3)*X + P(:,4));
    uv_hat = [uv_hat_tilde(1,:)./uv_hat_tilde(3,:); uv_hat_tilde(2,:)./uv_hat_tilde(3,:)];
    
    n = size(X,2);
    r = zeros(n*2, 1);
    for i = 1:n
       r((i - 1)*2 + 1:i*2) = uv(:,i) - uv_hat(:,i); 
    end
    
end