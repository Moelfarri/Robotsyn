function  [features,valid_points] =DetectAndExtractFeatures(J,method)
%J - Is a grayscaled/double image
%method - Type of feature extraction method

if method == "SURF"
    points = detectSURFFeatures(J);
    [features,valid_points] = extractFeatures(J,points);
    size(features)

elseif method == "ORB"
    %Increasing NumLevels give more matches - and higher time complexity
    %Reducing ROI helps 
    %Struggles during matching since it creates way too many features 
    
    %points = detectORBFeatures(J,'NumLevels',4,'ROI',[250 250 size(J,2)-500 size(J,1)-500]);
    points = detectORBFeatures(J,'NumLevels',18);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    

elseif method == "BRISK"
    %Lower Mincontrast gives more features
    points = detectBRISKFeatures(J,'MinContrast',0.01, 'MinQuality', 0.1);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    
    
elseif method == "KAZE"
    %Increasing NumScaleLevels between [3-10] gives more features
    points = detectKAZEFeatures(J,'NumScaleLevels',3);
    [features,valid_points] = extractFeatures(J,points);

    
    
else 
    points = -1;
    disp('Input Method does not exist. Try  SURF, ORB, BRISK OR KAZE');

end


end