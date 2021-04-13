data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
X = data.X;
FeatureDescriptor = data.FeatureDescriptor;
weights = false;

%New image comes in
I = imread('../our_own_data_images_and_figures\scene images/IMG_2493.JPEG');

% %undistort the image
[I, ~ ]  = undistortImage(I,params);

%convert to double and grescale
J = rgb2gray(im2double(I));

%detect SURF points
points = detectSURFFeatures(J);

%extract the neibourhood features
[features,valid_points] = extractFeatures(J,points);

%find the
indexPairs = matchFeatures(features,FeatureDescriptor);

%find 2D -3D correspondences
uv_i = double(valid_points(indexPairs(:,1),:).Location)';
X_matches = X(1:3,indexPairs(:,2));

%Find rough estiamtes of pose and an inlier set using MSAC (Similar to RANSAC)
[worldOrientation, worldLocation,inlierIdx] = estimateWorldCameraPose(uv_i',X_matches',params);

%Use nonlinear solver
P_i = [0, 0, 0, -worldLocation]';
R0 = worldOrientation;
func =@(P) lsqnonlin_func(K,R0,P,X_matches(:,inlierIdx),uv_i(:,inlierIdx),weights);

P_i = lsqnonlin(func, P_i);
R = return_R(R0, P_i);
t = P_i(4:6);

%Find the pose 
P_i = [R, t];
uv_i_tilde  = [uv_i; ones(1,size(uv_i,2))]; 
xy1 = inv(K)*uv1_tilde;
%Do Ransac to find the inlier set

    