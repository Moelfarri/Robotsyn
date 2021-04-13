function  [features,valid_points] =DetectAndExtractFeatures(J,method)
%J - Is a grayscaled/double image
%method - Type of feature extraction method

if method == "SURF"
    points = detectSURFFeatures(J);
    [features,valid_points] = extractFeatures(J,points);


elseif method == "ORB"
    %Struggles during matching since it creates way too many features 
    %Reducing NumLevels and choosing ROI helps
    %Use during modeling
    points = detectORBFeatures(J,'NumLevels',4,'ROI',[250 250 size(J,2)-500 size(J,1)-500]);
    
    %Use during Localization - can also be used if approximate matching is
    %used
    %points = detectORBFeatures(J,'NumLevels',18);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    

elseif method == "BRISK"
    %Lower Mincontrast gives more features
    points = detectBRISKFeatures(J,'MinContrast',0.12, 'MinQuality', 0.1);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    
    
elseif method == "KAZE"
    %Increasing NumScaleLevels between [3-10] gives more features
    points = detectKAZEFeatures(J,'NumScaleLevels',3);
    [features,valid_points] = extractFeatures(J,points);

elseif method == "FREAK"
    %Use BRISK feature detector as in the FREAK descriptor (Alahi et al. 2012)
    points = detectBRISKFeatures(J,'MinContrast',0.14);
    [features,valid_points] = extractFeatures(J,points,'Method','FREAK');
    features = features.Features;
    
    
else 
    points = -1;
    disp('Input Method does not exist. Try  SURF, ORB, BRISK, FREAK OR KAZE');

end


end