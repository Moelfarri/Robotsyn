function  [T_m2q, uv_inliers, X_inliers, jacobian] = localize(K,params,I,X,matchedFeatures2_inlier)

   
    % %undistort the images when using your own pictures
    [I, ~ ]  = undistortImage(I,params);
    
    %convert to double
    J = im2double(I);
    J = rgb2gray(J);
    points = detectSURFFeatures(J);
    
    %extract the neibourhood features
    [features,valid_points] = extractFeatures(J,points);

    indexPairs = matchFeatures(features,matchedFeatures2_inlier);
    
    matchedPoints = valid_points(indexPairs(:,1),:);
    uv = matchedPoints.Location; 
    

    [worldOrientation, worldLocation,inlierIdx] = estimateWorldCameraPose(double(uv),X(1:3,indexPairs(:,2))',params);
    
    X_matched = X(1:3,indexPairs(:,2));
    uv = uv';

    %transform without LM nonlin optimizer
    %T_m2q = [worldOrientation, -worldLocation';
    %         zeros(1,3), 1];
    
    %transform with LM nonlin optimizer
    %P_0 = [worldOrientation, -worldLocation'];
    P0 = [0 0 0 -worldLocation]; 
    R0 = worldOrientation;
    
    %3.1 Function:
    %func =@(P) lsqnonlin_func(K,R0,P,X_matched(:,inlierIdx),uv(:,inlierIdx));
    
    %3.3 Weighted Function:
    func =@(P) lsqnonlin_func_task3_3(K,R0,P,X_matched(:,inlierIdx),uv(:,inlierIdx));
    
    
    [pi,resnorm,residual,exitflag,output,lambda,jacobian] = lsqnonlin(func, P0);

    T_m2q = return_T(R0, pi(1), pi(2), pi(3), pi(4),pi(5), pi(6));
    
    
    

    
         
    
    X_inliers  = X_matched(:,inlierIdx);

    X_inliers  = [X_inliers; ones(1,size(X_inliers,2))];
    X_inliers = X_inliers(1:4,:);
    uv_inliers = uv(:,inlierIdx); 
    
end

