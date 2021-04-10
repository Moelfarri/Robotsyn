

data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;

I = imread('../our_own_data_images_and_figures\scene images/IMG_2441.JPEG');


[T_m2q, u_inliers, X_inliers, jacobian]  = localize(K,params,I,X,matchedFeatures2_inlier);

%3.2 Covariance calculations:
sigma_r = 1;
covariance_r = sigma_r^2 *eye(2*size(u_inliers,2));
covariance_p = inv(jacobian'*inv(covariance_r)*jacobian);
Std_cov_p    = sqrt(diag(covariance_p));



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