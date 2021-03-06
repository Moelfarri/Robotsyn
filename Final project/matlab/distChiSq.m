function feature_distances = distChiSq(features1, features2)

features1 = abs(features1);
features2 = abs(features2);
feature_distances = zeros(size(features1,1),size(features2,1));
for i = 1:size(features2,1)
    for j = 1:size(features1,1) 
    feature_distances(j,i) = sum((features2(i,:) - features1(j,:)).^2./(features2(i,:) + features1(j,:)));
    end
end
feature_distances = feature_distances/2;