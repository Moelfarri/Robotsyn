rand('seed',1);
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
X = data.X;
FeatureDescriptor = data.FeatureDescriptor;

%Set weights to true for task 3.3, otehrwise set it to false
weights = true;

% I = imread('../our_own_data_images_and_figures\scene images/IMG_2492.JPEG');
I = imread('../our_own_data_images_and_figures\scene images/IMG_2493.JPEG');
% I = imread('../our_own_data_images_and_figures\scene images/IMG_2500.JPEG');

%Find the least squares minimum pose and the inlier set
[P_i, X_i, uv_i] = localize(K,params,I,X,FeatureDescriptor,weights);
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

diag_Covariance_p_sqrt

