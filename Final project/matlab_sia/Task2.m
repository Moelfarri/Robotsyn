
%load the data
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;

% K = load('../hw5_data_ext/K.txt');
%for now only use pcitures included
I1 = imread('../our_own_data_images_and_figures\scene images/IMG_2445.JPEG');
I2 = imread('../our_own_data_images_and_figures\scene images/IMG_2443.JPEG');

% %undistort the images when using your own pictures
[I1, ~ ]  = undistortImage(I1,params);
[I2, ~ ]  = undistortImage(I2,params);


%convert to double
J1 = im2double(I1); 
J2 = im2double(I2);

J1 = rgb2gray(J1);
J2 = rgb2gray(J2);


points1 = detectSURFFeatures(J1);
points2 = detectSURFFeatures(J2);

%extract the neibourhood features
[features1,valid_points1] = extractFeatures(J1,points1);
[features2,valid_points2] = extractFeatures(J2,points2);

%match the features
indexPairs = matchFeatures(features1,features2);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);
matchedFeatures1 = features1(indexPairs(:,1),:);
matchedFeatures2 = features2(indexPairs(:,2),:);

% figure; ax = axes;
% showMatchedFeatures(J1,J2,matchedPoints1,matchedPoints2,'montage','Parent',ax);

uv1_tilde = [matchedPoints1.Location'; ones(1, size(matchedPoints1.Location,1))];
uv2_tilde = [matchedPoints2.Location'; ones(1, size(matchedPoints1.Location,1))];

xy1 = inv(K)*uv1_tilde;
xy2 = inv(K)*uv2_tilde;
num_trials  = get_num_ransac_trials(8, 0.99, 0.5);
[E,inliers] = estimate_E_ransac(xy1, xy2, K, 4, num_trials);
matchedFeatures1_inlier = matchedFeatures1(inliers,:);
matchedFeatures2_inlier = matchedFeatures2(inliers,:);

%find the innliers
xy1_inliers = xy1(:,inliers);
xy2_inliers = xy2(:,inliers);
uv1_tilde_inliers = uv1_tilde(:,inliers);
uv2_tilde_inliers = uv2_tilde(:,inliers);
uv2_inliers = [uv2_tilde_inliers(1,:)./uv2_tilde_inliers(3,:); uv2_tilde_inliers(2,:)./uv2_tilde_inliers(3,:)];
uv1_inliers = [uv1_tilde_inliers(1,:)./uv1_tilde_inliers(3,:); uv1_tilde_inliers(2,:)./uv1_tilde_inliers(3,:)];

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

% draw_point_cloud(X_image_1_tilde, im2double(I1), uv1_tilde_inliers, [-2,2], [-2,+2], [0,6]);
% draw_correspondences(I1, I2, uv1_tilde_inliers, uv2_tilde_inliers, F_from_E(E, K));

R_image_2 = closest_rotation_matrix(P2(1:3,1:3));

X_image_1 = X_image_1_tilde(1:3,:);


number_of_images = 2;
num_correpondences = size(uv2_tilde_inliers,2);

uv_cell = cell(number_of_images);
P_cell = cell(number_of_images);
R0_cell = cell(number_of_images);
uv_cell(1) = {uv1_inliers};
P_cell(1) = {zeros(6,1)};
R0_cell(1) = {eye(3,3)};
uv_cell(2) = {uv2_inliers};
P_cell(2) = {[zeros(1,3) P2(:,4)']'};
R0_cell(2) = {R_image_2};


[P,X] = LM(100, 1e-3, R0_cell, K , P_cell, X_image_1, uv_cell);


R_image_1 = return_R(R0_cell{1},P{1});
R_image_2 = return_R(R0_cell{2},P{2});

P_1 = [R_image_1,P{1}(4:6)];
P_2 = [R_image_2,P{2}(4:6)];

%change to camera corrdinates of image 1
X_image_1_tilde = [P_1;zeros(1,3),1]*[X;ones(1,size(X,2))];
draw_point_cloud(X_image_1_tilde, im2double(I1), uv1_tilde_inliers, [-2,2], [-2,+2], [0,6]);


%I should save the X, uv1 and uv2_inliers, and also maybe P1 and P2




