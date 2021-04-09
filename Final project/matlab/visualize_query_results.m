% This script uses example data. You will have to modify the loading code
% below to suit how you structure your data
%model = '../visualization_sample_data';
%query = '../visualization_sample_data/query/IMG_8210';
%K       = load(sprintf('%s/K.txt', model));       % Intrinsic matrix.
%X      = load(sprintf('%s/X.txt', model));       % 3D points [shape: 4 x num_points].
%T_m2q   = load(sprintf('%s_T_m2q.txt', query));   % Model-to-query transformation (produced by your localization script).
%matches = load(sprintf('%s_matches.txt', query)); % Initial 2D-3D matches (see usage code below).
%inliers = load(sprintf('%s_inliers.txt', query)); % Indices of inlier matches (see usage code below).
%u      = load(sprintf('%s_u.txt', query));       % Image location of features detected in query image (produced by your localization script). [shape: 2 x n].
%I       = imread(sprintf('%s.jpg', query));       % Query image.


K       = load(sprintf('%s/K.txt', model));       % Intrinsic matrix.
I       = imread("../visualization_sample_data/query/IMG_8207.jpg"); %Query image
I2      = imread("../visualization_sample_data/query/IMG_8210.jpg");

%convert to double and grayscale
I = im2double(I); 
I2 = im2double(I2);

I  = rgb2gray(I); 
I2 = rgb2gray(I2);



[X,T_m2q,u] = localize(I,I2,K);



%%%FUNKER IKKE??
%PnPSolver + Ransac algorithm:
focal_length   = [K(1,1) K(2,2)];
principalPoint = [K(1,3) K(2,3)];
imageRes       = [size(I,1) size(I,2)];
intrinsics = cameraIntrinsics(focal_length,principalPoint,imageRes); 

imagePoints = double(xy1(1:2,:)');
worldPoints = X(1:3,:)';
[worldOrientation, worldLocation,inliers] = estimateWorldCameraPose(imagePoints,worldPoints,intrinsics); 
[rotationMatrix, translationVector] = cameraPoseToExtrinsics(worldOrientation, worldLocation);

t_x =[0 -translationVector(3) translationVector(2) ; translationVector(3) 0 -translationVector(1) ; -translationVector(2) translationVector(1) 0 ];
E = t_x*rotationMatrix;
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
        best_T = T;
    end
end
%X = best_X;
%T_m2q = best_T; 




assert(size(X,1) == 4);
assert(size(u,1) == 2);

% If you have colors for your point cloud model, then you can use this.
%c = load(sprintf('%s/c.txt', model)); % RGB colors [shape: num_points x 3].
% Otherwise you can use this, which colors the points according to their Y.
c = [];

% These control the location and the viewing target of the virtual figure
% camera, in the two views. You will probably need to change these to work
% with your scene.
lookfrom1 = [0 -20 5]';
lookat1   = [0 0 6]';
lookfrom2 = [20 0 8]';
lookat2   = [0 0 8]';

% You may want to change these too.
point_size = 3;
frame_size = 0.5; % Length of visualized camera axes.

% NB! I generated the sample data in Python, which uses 0-based indexing.
% To get 1-based indexing I have to add 1. You probably not need to do
% this if you generate your data in Matlab, so you will want to comment
% these lines out!

%matches = matches + 1;
%inliers = inliers + 1;

% 'matches' is assumed to be a Nx2 array, where the first column is the
% index of the 2D point among the query features and the second column is
% the index of its matched 3D point in the model (X).

%Hvordan skal de brukes?
%u_matches = u(:,matches(:,1));
%X_matches = X(:,matches(:,2));

% 'inliers' is assumed to be a 1D array of indices of the good matches,
% e.g. as identified by your PnP+RANSAC strategy.
u_inliers = u; %u_matches(:,inliers);
X_inliers = X;%X_matches(:,inliers);

u_hat = project(K, T_m2q*X_inliers);
e = vecnorm(u_hat - u_inliers);

figure(1);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.6, 0.8]);

subplot(221);
imagesc(I); hold on;
scatter(u_hat(1,:), u_hat(2,:), 10, e, 'filled');
axis equal;
axis ij;
xlim([0, size(I,2)]);
ylim([0, size(I,1)]);
cbar = colorbar;
cbar.Label.String = 'Reprojection error (pixels)';
title('Query image and reprojected points');

subplot(222);
histogram(e, 'NumBins', 50);
xlabel('Reprojection error (pixels)');

subplot(223);
draw_model_and_query_pose(X, T_m2q, K, lookat1, lookfrom1, point_size, frame_size, c);
title('Model and localized pose (top view)');

subplot(224);
draw_model_and_query_pose(X, T_m2q, K, lookat2, lookfrom2, point_size, frame_size, c);
title('Model and localized pose (side view)');
