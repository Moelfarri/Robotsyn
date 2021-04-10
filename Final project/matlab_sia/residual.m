function r = residual(R0,P,X,uv,K,all)
    %Given a vector of 3D world points and given P
    %the function will create a residual residual vector r: 2*n x 1
    %correspondences

    if all == 2
        num_images = size(uv,1);
        num_correpondences = size(uv{1},2);
        
        total_num_residuals = 2*num_correpondences*num_images;
        r = zeros(total_num_residuals,1);
        for i =1:num_images
           r((i - 1)*2*num_correpondences + 1:i*2*num_correpondences) = residual(R0{i},P{i},X,uv{i},K,1);
        end
    
    elseif all == 1
        R = return_R(R0,P);
        t = P(4:6);
        n = size(uv,2);
        uv_hat_tilde = K*(R * X + t);
        uv_hat = [uv_hat_tilde(1,:)./uv_hat_tilde(3,:); uv_hat_tilde(2,:)./uv_hat_tilde(3,:)];
        r = zeros(2*n,1);
        for i = 1:n
            r((i-1)*2 + 1 : i*2) = uv_hat(:,i) - uv(:,i);
        end
    else 
        R = return_R(R0,P);
        t = P(4:6);
        uv_hat_tilde = K*(R * X + t);
        uv_hat = [uv_hat_tilde(1,:)/uv_hat_tilde(3,:); uv_hat_tilde(2,:)/uv_hat_tilde(3,:)];
        r = uv_hat - uv;
    end
    
end


