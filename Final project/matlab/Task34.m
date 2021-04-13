
data = load('../our_own_data_images_and_figures/data');
K = data.K;
params = data.params;
X = data.X;
FeatureDescriptor = data.FeatureDescriptor;
estimationErrors = data.estimationErrors;
weights = false;
m = 500;


I = imread('../our_own_data_images_and_figures\scene images/IMG_2492.JPEG');

% %scnario a 
% sigma_f = 50;
% sigma_x = 0.1;
% sigma_y = 0.1;

% scnario b
% sigma_f = 0.1;
% sigma_x = 50;
% sigma_y = 0.1;
% 
% scenario c
sigma_f = 0.1;
sigma_x = 0.1;
sigma_y = 50;


Covariance_P = MonteCarlo(params,I,K, X,FeatureDescriptor,m,sigma_f,sigma_x,sigma_y);


%Find the diagonal entries and the square root
Covariance_P = diag(Covariance_P);
Covariance_P = sqrt(Covariance_P(:));

%Convert to degrees and mm
Covariance_P(1:3) = Covariance_P(1:3)*180/pi;

Covariance_P(4:6) = Covariance_P(4:6)*10^3;

Covariance_P
