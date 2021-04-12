%Task 1.2
data = load('../our_own_data_images_and_figures/data');
estimationErrors = data.estimationErrors;
params = data.params;
I = imread('../our_own_data_images_and_figures/cam images/IMG_2346.JPEG');

%inrisic parmeters and lens distortions
FocalLength = params.FocalLength;
PrincipalPoint = params.PrincipalPoint;
RadialDistortion = params.RadialDistortion;
TangentialDistortion = params.TangentialDistortion;
Skew = params.Skew;

%errors (skew and Tangential error has a standard deviation of 0, therefore
%they don't need to be sampled)
FocalLengthError = estimationErrors.IntrinsicsErrors.FocalLengthError;
PrincipalPointError = estimationErrors.IntrinsicsErrors.PrincipalPointError;
RadialDistortionError = estimationErrors.IntrinsicsErrors.RadialDistortionError;


FocalLength_temp = FocalLength;
PrincipalPoint_temp = PrincipalPoint;
RadialDistortion_temp = RadialDistortion;

figure;
imshow(I);
title('original image');
for i = 1:10
    %pull samples from a normal distribution
    FocalLength_temp(1) = normrnd(FocalLength(1), FocalLengthError(1));
    FocalLength_temp(2) = normrnd(FocalLength(2), FocalLengthError(2));
    PrincipalPoint_temp(1) = normrnd(PrincipalPoint(1),PrincipalPointError(1));
    PrincipalPoint_temp(2) = normrnd(PrincipalPoint(2),PrincipalPointError(2));
    RadialDistortion_temp(1) = normrnd(RadialDistortion(1),RadialDistortionError(1));
    RadialDistortion_temp(2) = normrnd(RadialDistortion(2),RadialDistortionError(2));
    
    IntrinsicMatrix = [FocalLength_temp(1), Skew,  PrincipalPoint_temp(1);
                        0, FocalLength_temp(2), PrincipalPoint_temp(2);
                        0, 0, 1];
    cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix','RadialDistortion',RadialDistortion_temp,'TangentialDistortion',TangentialDistortion );

   
    figure;
    [J, newOrigin] = undistortImage(I,cameraParams);
    imshow(J);
    title('one of the undistorted images');
end


%Comment
%Except one or two of the images, the the diference between the original
%image and the undistorted is very difficult to see
%The distortion is very subtle and if it werent  for the fact that we knew
%the images were different, i don't think we would see the difference. 
%Overall i don't think we need our camera images to be undistorted.





