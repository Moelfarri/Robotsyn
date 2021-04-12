function [P_i , X_new, uv_new] =  localize(K,params,I,X,featureDescriptor,weights)

    % %undistort the images when using your own pictures
    [I, ~ ]  = undistortImage(I,params);
    
    %convert to double
    J = rgb2gray(im2double(I));
    
    %extract the neibourhood features 
    %Note model from task2 should have same feature detector as in this
    %task.
    [features,valid_points] = DetectAndExtractFeatures(J,"ORB");
    

    indexPairs = matchFeatures(features,featureDescriptor);
    
    matchedPoints = valid_points(indexPairs(:,1),:);
    uv = matchedPoints.Location';
    X_matches = X(1:3,indexPairs(:,2));
    

    [worldOrientation, worldLocation,inlierIdx] = estimateWorldCameraPose(double(uv'),X_matches',params);
    
    P_i = [0, 0, 0, -worldLocation]';
    
    R0 = worldOrientation;
    func =@(P) lsqnonlin_func(K,R0,P,X_matches(:,inlierIdx),uv(:,inlierIdx),weights);
    
    P_i = lsqnonlin(func, P_i);
    R = return_R(R0, P_i);
    t = P_i(4:6);
    P_i = [R, t];
    X_new = X_matches(:,inlierIdx);
    uv_new = uv(:,inlierIdx);
    
    inlieridx = size(inlierIdx,1)
    inlieridxwouthoutliers = size(X_new,2)

end

