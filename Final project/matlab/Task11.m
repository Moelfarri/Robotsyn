
%create a set of calibration images
images = imageDatastore('../our_own_data_images_and_figures/cam images');
imageFileNames = images.Files;

%detect calibration patters
[imagePoints, boardsize] = detectCheckerboardPoints(imageFileNames);

%generate world coordinates of the corners of the squares
squareSize = 15;
worldPoints = generateCheckerboardPoints(boardsize, squareSize);

%calibrae the camera
I = readimage(images, 1);
imageSize = [size(I,1), size(I,2)];
[params, ~ , estimationErrors] = estimateCameraParameters(imagePoints, worldPoints,'ImageSize', imageSize);  


%Extrinsic parameters visualization (we don't need this)
% figure; 
% showExtrinsics(params, 'PatternCentric');

%Task 1.1 a
%mean reproection error per Image
figure; 
showReprojectionErrors(params); 

%2D scatter plot of the reprojection errors
figure; 
showReprojectionErrors(params, 'scatterPlot'); 

%Task 1.1 b
%standard deviation of the intrinsic parameters
displayErrors(estimationErrors,params);

K = params.IntrinsicMatrix';
save('../our_own_data_images_and_figures/data','K','params');
 

%when you want to read the file, do the following
% data = load('../our_own_data_images_and_figures/data');
% K = data.K;
% params = data.params


