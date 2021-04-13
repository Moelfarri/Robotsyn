function [P_i , X_new, uv_new] =  localize(K,params,I,X,featureDescriptor,weights)

    % %undistort the image
    [I, ~ ]  = undistortImage(I,params);
    
    %convert to double and grescale
    J = rgb2gray(im2double(I));
    
    %detect SURF points
    points = detectSURFFeatures(J);
    
    %extract the neibourhood features
    [features,valid_points] = extractFeatures(J,points);

    %find the 
    indexPairs = matchFeatures(features,featureDescriptor);
    
    %find 2D -3D correspondences
    uv = double(valid_points(indexPairs(:,1),:).Location)';
    X_matches = X(1:3,indexPairs(:,2));
    
    %Find rough estiamtes of pose and an inlier set using MSAC (Similar to RANSAC)
    [worldOrientation, worldLocation,inlierIdx] = estimateWorldCameraPose(uv',X_matches',params);
    
    %Use nonlinear solver
    P_i = [0, 0, 0, -worldLocation]';
    R0 = worldOrientation;
    func =@(P) lsqnonlin_func(K,R0,P,X_matches(:,inlierIdx),uv(:,inlierIdx),weights);
    
    P_i = lsqnonlin(func, P_i);
    R = return_R(R0, P_i);
    t = P_i(4:6);
    
    %Return pose, the 3D inlier set, and the uv inlier set
    P_i = [R, t];
    X_new = X_matches(:,inlierIdx);
    uv_new = uv(:,inlierIdx);
    
end

