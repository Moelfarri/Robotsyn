function covariance_p = MonteCarloCovarianceEstimator(params,estimationErrors,X_inliers,uv_inliers,RoughCameraEstimate)
    m = 500;
    P_LIST = zeros(m,6);
    R0 = RoughCameraEstimate(1:3,1:3);
    t0 = RoughCameraEstimate(1:3,4);
    P0 = [0 0 0 -t0']; 
    X_inliers = X_inliers(1:3,:);
    
    FocalLength = params.FocalLength;
    PrincipalPoint = params.PrincipalPoint;
    RadialDistortion = params.RadialDistortion;
    Skew = params.Skew;

    FocalLengthError = estimationErrors.IntrinsicsErrors.FocalLengthError;
    PrincipalPointError = estimationErrors.IntrinsicsErrors.PrincipalPointError;
    RadialDistortionError = estimationErrors.IntrinsicsErrors.RadialDistortionError;

    for i = 1:m
    FocalLength_temp(1) = normrnd(FocalLength(1), FocalLengthError(1));
    FocalLength_temp(2) = normrnd(FocalLength(2), FocalLengthError(2));
    PrincipalPoint_temp(1) = normrnd(PrincipalPoint(1),PrincipalPointError(1));
    PrincipalPoint_temp(2) = normrnd(PrincipalPoint(2),PrincipalPointError(2));
    RadialDistortion_temp(1) = normrnd(RadialDistortion(1),RadialDistortionError(1));
    RadialDistortion_temp(2) = normrnd(RadialDistortion(2),RadialDistortionError(2));
    
    K  = [FocalLength_temp(1), Skew,  PrincipalPoint_temp(1);
                        0, FocalLength_temp(2), PrincipalPoint_temp(2);
                        0, 0, 1];
    
    %Estimate Pose with the same inlier set:
    func =@(P) lsqnonlin_func(K,R0,P,X_inliers,uv_inliers);
    pi = lsqnonlin(func, P0);
     
    
    %Append estimated parameters into a list
    P_LIST(i,:) = pi';
    end
    covariance_p = cov(P_LIST); 
end