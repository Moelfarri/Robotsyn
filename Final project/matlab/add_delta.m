function [P_delta,X_delta] = add_delta(P_cell,X,delta)
    num_images = size(P_cell,1);
    delta_P = delta(1:num_images*6);
    delta_X = delta(num_images*6 + 1:end);
    
    num_X_variables = size(delta_X,1);
    num_correspondences = num_X_variables/3; 
    delta_X = reshape(delta_X,3,num_correspondences);
    
    X_delta = X + delta_X;
    
    P_delta = cell(num_images,1);
    for i = 1:num_images
        P_delta(i) = {P_cell{i} + delta_P((i - 1)*6 + 1:i*6)};
        
    end

end

