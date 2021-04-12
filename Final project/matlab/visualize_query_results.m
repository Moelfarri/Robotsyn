% This script uses example data. You will have to modify the loading code
% below to suit how you structure your data.
rand('seed',1);
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
X = data.X;
FeatureDescriptor = data.FeatureDescriptor;


I = imread('../our_own_data_images_and_figures\scene images/IMG_2442.JPEG');
[P_i, X_i, u] = localize(K,params,I,X,FeatureDescriptor,true);

X_i = [X_i;ones(1,size(X_i,2))];
T_m2q = [P_i;zeros(1,3), 1];



assert(size(X_i,1) == 4);
assert(size(u,1) == 2);
c = [];
% with your scene.
lookfrom1 = [0 -20 5]';
lookat1   = [0 0 6]';
lookfrom2 = [20 0 8]';
lookat2   = [0 0 8]';

% You may want to change these too.
point_size = 3;
frame_size = 0.5; % Length of visualized camera axes.
u_hat = project(K, T_m2q*X_i);
e = vecnorm(u_hat - u);


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
draw_model_and_query_pose(X_i, T_m2q, K, lookat1, lookfrom1, point_size, frame_size, c);
title('Model and localized pose (top view)');

subplot(224);
draw_model_and_query_pose(X_i, T_m2q, K, lookat2, lookfrom2, point_size, frame_size, c);
title('Model and localized pose (side view)');
