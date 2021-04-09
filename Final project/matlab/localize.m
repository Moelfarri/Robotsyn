function [X,T_m2q,u] = localize(I,I2,K)

points1 = detectSURFFeatures(I);
points2 = detectSURFFeatures(I2);

%extract the neibourhood features same as descriptors in opencv
[features1,valid_points1] = extractFeatures(I,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

%match the features
indexPairs = matchFeatures(features1,features2);



matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);


 
%2D pixel correspondences via descriptor matching:
uv1 = [matchedPoints1.Location'; ones(1, size(matchedPoints1.Location,1))];
uv2 = [matchedPoints2.Location'; ones(1, size(matchedPoints1.Location,1))];



% Image location of features detected in query image (produced by your localization script). [shape: 2 x n].
u = matchedPoints1.Location';

xy1 = inv(K)*uv1;
xy2 = inv(K)*uv2;


%use RANSAC to get rough camera pose estimate and an inlier set:
num_trials  = get_num_ransac_trials(8, 0.99, 0.5);
[E,inliers] = estimate_E_ransac(xy1, xy2, K, 4, num_trials);

xy1_inliers = xy1(:,inliers);
xy2_inliers = xy2(:,inliers);
uv1_inliers = uv1(:,inliers);
uv2_inliers = uv2(:,inliers);

P1 = [eye(3), zeros(3,1)];

T_all = decompose_E(E);
best_num_visible = 0;
for i=1:4
    T = T_all(:,:,i);
    P2 = T(1:3,:);
    X1 = triangulate_many(xy1, xy2, P1, P2);
    X2 = T*X1;
    num_visible = sum((X1(3,:) > 0) & (X2(3,:) > 0));
    if num_visible > best_num_visible
        best_num_visible = num_visible;
        best_i = i;
        best_X = X1;
    end
end

%3D real world correspondences and camera pose
X = best_X;
T_m2q = T_all(:,:,best_i);

%Nonlinear optimizer to better estimate camera-pose 
t  = T_m2q(1:3,4);
p  = [0 0 0 t'];
R0 = T_m2q(1:3,1:3);
residualsfun = @(p) residuals(X,K,u,R0,p(1), p(2), p(3), p(4), p(5), p(6));
p = levenberg_marquardt(residualsfun,p,1e-6);

T_m2q = return_T(R0,p(1), p(2), p(3), p(4), p(5), p(6));

end 