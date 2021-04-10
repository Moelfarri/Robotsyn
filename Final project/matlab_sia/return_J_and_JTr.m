function [J, JTr] = return_J_and_JTr(num_images, num_correpondences, R_cell, K , P_cell, X, uv_cell)
    %Cam params and feature params per image
    camera_params = 6;
    feature_params = 3*num_correpondences;
    
    %Number of residuals per image
    num_residuals = 2*num_correpondences;
    
    
    J = zeros(num_residuals*num_images,6*num_images + feature_params);
    JTr = zeros(6*num_images + feature_params,1);
    
    
    A = zeros(num_residuals,camera_params);
    B = zeros(num_residuals, feature_params);
    epsilon = 1e-5;     
    Epsilon_pose = diag(epsilon*ones(6,1));
    Epsilon_corres = diag(epsilon*ones(3,1));
    
    J_pose_temp_left = zeros(num_residuals,1);
    J_pose_temp_right = zeros(num_residuals,1);
    
    J_X_temp_left = zeros(2,1);
    J_X_temp_right = zeros(2,1);
    
    
    for immage = 1:num_images
        P = P_cell{immage};
%         X = X_cell{immage};
        uv = uv_cell{immage};
        R0 = R_cell{immage};
        for i = 1:6
            J_pose_temp_left(:) = residual(R0,P + Epsilon_pose(:,i), X, uv, K, 1);
            J_pose_temp_right(:) = residual(R0, P - Epsilon_pose(:,i), X, uv,K, 1);
            A(:,i) = (J_pose_temp_left - J_pose_temp_right)/(2*epsilon);
           
            JTr((immage - 1)*6 + i) = A(:,i).'*residual(R0,P, X, uv, K, 1);
        end
        for i = 1:num_correpondences
            for j = 1:3
                J_X_temp_left(:) = residual(R0, P, X(:,i) + Epsilon_corres(:,j), uv(:,i),K, 0);
                J_X_temp_right(:) = residual(R0,P, X(:,i) - Epsilon_corres(:,j), uv(:,i),K, 0);
                B((i - 1)*2 + 1 : i*2 , 3*(i - 1) + j) = (J_X_temp_left - J_X_temp_right)/(2*epsilon);
                
                JTr(6*num_images + (i - 1)*3 + j) = JTr(6*num_images + (i - 1)*3 + j) + B((i - 1)*2 + 1 : i*2 ,3*(i - 1) + j).'*residual(R0,P, X(:,i), uv(:,i),K, 0);
            end
        end
        
        J((immage - 1)*num_residuals + 1:immage * num_residuals, (immage - 1)*6 + 1 : immage*6) = A;
        
        J((immage - 1)*num_residuals + 1:immage * num_residuals, camera_params*num_images + 1:end) = B;
         
    end
end


