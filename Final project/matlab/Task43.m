%load the data
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
estimationErrors = data.estimationErrors;

I1 = imread('../our_own_data_images_and_figures\scene images/IMG_2496.JPEG');
I2 = imread('../our_own_data_images_and_figures\scene images/IMG_2497.JPEG');

%Choose one of the five Descriptor methods here
%SURF,KAZE,ORB, BRISK or FREAK into the method variable
%NOTE YOU MIGHT GET "OUT OF MEMORY" ERROR, change to approximate setting
%on the matcher (line 35)
method = 'ORB';
%%
%Modeling With different feature descriptors 

% %undistort the images when using your own pictures
[I1, ~ ]  = undistortImage(I1,params);
[I2, ~ ]  = undistortImage(I2,params);

%convert to double
J1 = rgb2gray(im2double(I1)); 
J2 = rgb2gray(im2double(I2));

%4.3 - 1 of the five feature descriptors
[features1,valid_points1] = DetectAndExtractFeatures(J1,method);
[features2,valid_points2] = DetectAndExtractFeatures(J2,method);


%match the features
indexPairs = matchFeatures(features1,features2);

%Alternative matching for big featuresets in ORB/FREAK/BRISK
%indexPairs = matchFeatures(features1,features2, 'Method', 'Approximate');

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);
FeatureDescriptor = features1(indexPairs(:,1),:);

% figure; ax = axes;
% showMatchedFeatures(J1,J2,matchedPoints1,matchedPoints2,'montage','Parent',ax);

uv1_tilde = [matchedPoints1.Location'; ones(1, size(matchedPoints1.Location,1))];
uv2_tilde = [matchedPoints2.Location'; ones(1, size(matchedPoints1.Location,1))];

%Cross checking with E RANSAC 
rand('seed',1);
xy1 = inv(K)*uv1_tilde;
xy2 = inv(K)*uv2_tilde;
num_trials  = get_num_ransac_trials(8, 0.99, 0.5);
[E,inliers] = estimate_E_ransac(xy1, xy2, K, 4, num_trials);
FeatureDescriptor = FeatureDescriptor(inliers,:);

%find the innliers
xy1_inliers = xy1(:,inliers);
xy2_inliers = xy2(:,inliers);
uv1_tilde_inliers = uv1_tilde(:,inliers);
uv2_tilde_inliers = uv2_tilde(:,inliers);
uv2_inliers = [uv2_tilde_inliers(1,:)./uv2_tilde_inliers(3,:); uv2_tilde_inliers(2,:)./uv2_tilde_inliers(3,:)];
uv1_inliers = [uv1_tilde_inliers(1,:)./uv1_tilde_inliers(3,:); uv1_tilde_inliers(2,:)./uv1_tilde_inliers(3,:)];

%Find the Pose and a rough estiamte of 3D model
P1 = [eye(3), zeros(3,1)];
tot = 0;
T_all = decompose_E(E);
for i = 1:4
   R = T_all(1:3,1:3,i);
   t = T_all(1:3,4,i);
   X_image_1_tilde = triangulate_many(xy1_inliers, xy2_inliers, P1, [R, t]);
   X_image_2_tilde = T_all(:,:,i)*X_image_1_tilde;
   if sum((X_image_2_tilde(3,:) >= 0) & (X_image_1_tilde(3,:) > 0)) > tot
       tot = sum((X_image_2_tilde(3,:) >= 0) & (X_image_1_tilde(3,:) > 0));
       P2 = [R, t];
   end
end
X_image_1_tilde = triangulate_many(xy1_inliers, xy2_inliers, P1, P2);
R_image_2 = closest_rotation_matrix(P2(1:3,1:3));

%Put data in cells
uv_cell = cell(2);
P_cell = cell(2);
R0_cell = cell(2);

uv_cell(1) = {uv1_inliers};
uv_cell(2) = {uv2_inliers};

P_cell(1) = {zeros(6,1)};
P_cell(2) = {[zeros(1,3) P2(:,4)']'};

R0_cell(1) = {eye(3,3)};
R0_cell(2) = {R_image_2};



%Run LM
[P,X] = Levenberg_Marquardt(50, 1e-3, R0_cell, K , P_cell, X_image_1_tilde(1:3,:), uv_cell);

%Extract poses
R_image_1 = return_R(R0_cell{1},P{1});
R_image_2 = return_R(R0_cell{2},P{2});

P_1 = [R_image_1,P{1}(4:6)];
P_2 = [R_image_2,P{2}(4:6)];


%Reprojection Errors, :
u = uv1_inliers;
X_image_1_tilde = [P_1;zeros(1,3),1]*[X;ones(1,size(X,2))];
u_hat = project(K, X_image_1_tilde);
e = vecnorm(u_hat - u);

disp("Modeling Data for 4.3:")
inliers_and_outliers = size(inliers,2)
inlier_corrspondences_only = size(uv1_inliers,2)
min_e  = min(e)
mean_e = mean(e)
max_e  = max(e)




%%
%Different feature descriptors during localization 
%Note you have to run the same descriptor in both parts 
%Choose feature descriptor inside the localize43 function at the bottom
I = imread('../our_own_data_images_and_figures\scene images/IMG_2492.JPEG');
rand('seed',1);
weights = false;

if strcmp(method,'ORB')
    method = "ORB-LOCALIZATION";
end

[P_i, X_i, uv_i] = localize43(K,params,I,X,FeatureDescriptor,weights,method);



p = [0, 0, 0, P_i(:,4)']';
%comptue J
epsilon = 1e-5;
Epsilon = diag(epsilon*ones(6,1));
n = size(uv_i, 2); 
J = zeros(2*n,6);
%compute J
for i = 1:6   
   J_temp_right =  lsqnonlin_func(K,P_i(1:3,1:3),p + Epsilon(:,i),X_i,uv_i,weights);
   J_temp_left =  lsqnonlin_func(K,P_i(1:3,1:3),p - Epsilon(:,i),X_i,uv_i,weights);
   J(:,i) = (J_temp_right - J_temp_left)/(2*epsilon);
end

%Compute Covaraince matrix
sigma_r  = 1; 
Covariance_r = sigma_r *eye(2*n);
Covariance_p = (J.'*(Covariance_r)^(-1)*J)^(-1);

%Find the diagonal entries and the square root
diag_Covariance_p = diag(Covariance_p);
diag_Covariance_p_sqrt= sqrt(diag_Covariance_p(:));

%Convert to degrees and mm
diag_Covariance_p_sqrt(1:3) = diag_Covariance_p_sqrt(1:3)*180/pi;

diag_Covariance_p_sqrt(4:6) = diag_Covariance_p_sqrt(4:6)*10^3;






%PRINT DATA:
disp("Localization Data for 4.3:");
u = uv_i;
u_hat = project(K, [P_i;zeros(1,3),1]*[X_i;ones(1,size(X_i,2))]);
e = vecnorm(u_hat - u);

min_e  = min(e)
mean_e = mean(e)
max_e  = max(e)
diag_Covariance_p_sqrt



function [P_i , X_new, uv_new] =  localize43(K,params,I,X,featureDescriptor,weights,method)

    % %undistort the image
    [I, ~ ]  = undistortImage(I,params);
    
    %convert to double and grescale
    J = rgb2gray(im2double(I));
    
    
    %4.3 - one of the 5 feature descriptors
    [features,valid_points] = DetectAndExtractFeatures(J,method);
    

    %find the 
    indexPairs1 = matchFeatures(features,featureDescriptor);
    
    %find 2D -3D correspondences
    uv = double(valid_points(indexPairs1(:,1),:).Location)';
    X_matches = X(1:3,indexPairs1(:,2));
    
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
    
    %Print inliers
    inliers_and_outliers = size(inlierIdx,1)
    inlier_corrspondences_only = size(X_new,2)

end


