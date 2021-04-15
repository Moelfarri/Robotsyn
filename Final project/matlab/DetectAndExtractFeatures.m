function  [features,valid_points] =DetectAndExtractFeatures(J,method)
%J - Is a grayscaled/double image
%method - Type of feature extraction method

if strcmp(method,'SURF')
    points = detectSURFFeatures(J);
    [features,valid_points] = extractFeatures(J,points);


elseif strcmp(method,'ORB')
    %Struggles during matching since it creates way too many features 
    %Reducing NumLevels and choosing ROI helps & selectStrongest method is
    %also a possibility - used during modeling 
    %points = detectORBFeatures(J,'NumLevels',18); %
    %points = points.selectStrongest(40000);
    points = detectORBFeatures(J,'NumLevels',4,'ROI',[250 250 size(J,2)-500 size(J,1)-500]);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
   
elseif strcmp(method,'ORB-LOCALIZATION')
    %Use during Localization of ORB
    points = detectORBFeatures(J,'NumLevels',8);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    

elseif strcmp(method,'BRISK')
    %Lower Mincontrast gives more features
    points = detectBRISKFeatures(J,'MinContrast',0.12, 'MinQuality', 0.1);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    
    
elseif strcmp(method,'KAZE')
    %Increasing NumScaleLevels between [3-10] gives more features
    points = detectKAZEFeatures(J,'NumScaleLevels',3);
    [features,valid_points] = extractFeatures(J,points);

elseif strcmp(method,'FREAK')
    %Use BRISK feature detector as in the FREAK descriptor (Alahi et al. 2012)
    points = detectBRISKFeatures(J,'MinContrast',0.14);
    [features,valid_points] = extractFeatures(J,points,'Method','FREAK');
    features = features.Features;
    
    
else 
    points = -1;
    disp('Input Method does not exist. Try  SURF, ORB, BRISK, FREAK OR KAZE');

end


end