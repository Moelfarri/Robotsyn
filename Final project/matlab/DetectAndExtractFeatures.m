function  [features,valid_points] =DetectAndExtractFeatures(J,method)
%J - Is a grayscaled/double image
%method - Type of feature extraction method

if method == "SURF"
    points = detectSURFFeatures(J);
    [features,valid_points] = extractFeatures(J,points);

elseif method == "ORB"
    points = detectORBFeatures(J,'NumLevels',1);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;

elseif method == "BRISK"
    points = detectBRISKFeatures(J);
    [features,valid_points] = extractFeatures(J,points);
    features = features.Features;
    

else 
    points = -1;
    
end


end