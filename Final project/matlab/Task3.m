rand('seed',1);
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
X = data.X;
FeatureDescriptor = data.FeatureDescriptor;
estimationErrors = data.estimationErrors;
weights = false;
%Task 3.2 and 3.3
I = imread('../our_own_data_images_and_figures\scene images/IMG_2492.JPEG');
[P_i, X_i, uv_i] = localize(K,params,I,X,FeatureDescriptor,weights);
R0 = P_i(1:3,1:3);
t = P_i (:,4);
p = [0, 0, 0, t']';


%comptue J
epsilon = 1e-5;
Epsilon = diag(epsilon*ones(6,1));
n = size(uv_i, 2); 
J = zeros(2*n,6);
for i = 1:6   
   J_temp_right =  lsqnonlin_func(K,R0,p + Epsilon(:,i),X_i,uv_i,weights);
   J_temp_left =  lsqnonlin_func(K,R0,p - Epsilon(:,i),X_i,uv_i,weights);
   J(:,i) = (J_temp_right - J_temp_left)/(2*epsilon);
end

%Compute Covaraince matrix
sigma_r  = 1; 
Covariance_r = sigma_r *eye(2*n);
Covariance_p = (J.'*(Covariance_r)^(-1)*J)^(-1);
diag_Covariance_p = diag(Covariance_p);
diag_Covariance_p_squared = sqrt(diag_Covariance_p(:));

%Convert to degrees and mm
diag_Covariance_p_squared(1:3) = diag_Covariance_p_squared(1:3)*180/pi;
diag_Covariance_p_squared(4:6) = diag_Covariance_p_squared(4:6)*10^3;



%Task 3.4
%will monte carlo here so the imae needs to be given as an imput here
m = 500;
sigma_f = 50;
sigma_x = 0.1;
sigma_y = 0.1;
%Covariance_P = MonteCarlo(params,I,K, X,FeatureDescriptor,m,sigma_f,sigma_x,sigma_y);



%FOR TASK 4.3:
u = uv_i;
u_hat = project(K, [P_i;zeros(1,3),1]*[X_i;ones(1,size(X_i,2))]);
e = vecnorm(u_hat - u);
%inlier_corrspondences = size(uv1_inliers,2)
mean_e = mean(e)
max_e  = max(e)
min_e  = min(e)


%UNCERTAINTY OF THE POSE:
diag_Covariance_p_squared

