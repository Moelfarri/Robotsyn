%load the data
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
estimationErrors = data.estimationErrors;

I1 = imread('../our_own_data_images_and_figures\scene images/IMG_2496.JPEG');
I2 = imread('../our_own_data_images_and_figures\scene images/IMG_2497.JPEG');

%CHOOSE DISTANCE METRIC HERE: EUCLIDEAN, HELLINGER OR CHISQUARED
metric = 'EUCLIDEAN';
%%
%Modeling With different feature descriptors 

% %undistort the images when using your own pictures
[I1, ~ ]  = undistortImage(I1,params);
[I2, ~ ]  = undistortImage(I2,params);

%convert to double
J1 = rgb2gray(im2double(I1)); 
J2 = rgb2gray(im2double(I2));

%points1 =  detectSURFFeatures(J1);
%points2 =  detectSURFFeatures(J2);


%extract the neibourhood features
%[features1,valid_points1] =  extractFeatures(J1,points1);
%[features2,valid_points2] =  extractFeatures(J2,points2);

[valid_points1,features1] = vl_sift(J1);
[valid_points2,features2] = vl_sift(J1);


%Task 4.4 match Nearest Neighbor features with choosen distance
indexPairs = NNmatcher(features1,features2,metric);  


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

disp("Modeling Data for 4.4:")
inliers_and_outliers = size(inliers,2)
inlier_corrspondences_only = size(uv1_inliers,2)
mean_e = mean(e)





