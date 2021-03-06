function indexPairs = NNmatcher(features1, features2,metric)
assert(size(features1,2) == size(features2,2))
%Input: 
%      features1 : n_1 x m 
%      features2 : n_2 x m
%      metric    : EUCLIDEAN, Chi-Squared, Hellinger
%      Note that n_1 can be the same as n_2 but not necessarily
%      features1 and features2 are SURF features detected by the
%      SURF detection algorithm of two different images.
%Output: 
%     indexPairs : n_3 x 2 
%     where n_3 is the number of matches and the columns represent
%     indices in features1 and features2 respectively.

%The algorithm simply calculates all distances between features1 and 2
%Then sorts all the distances by row and keeps the indices of pre-sorted
%features. After that we find the 2 nearest neighbors, the nearest neighbor
%will be the first column of the sorted feature distances whilst the second
%column represents the second nearest neighbors. These are then divided by
%each other to obtain the Nearest Neighbor Distance Ratio (NNDR), the lower 
%the NNDR the better the match. The idea
%is to identify non-related matches by having an acceptance_threshold
%and then returning the relevant indexpair matches based on this threshold.


%Defining an acceptance threshold - the lower it is the more restrictive it 
%is on what is defined as matched features (SURF paper defines this to 0.7)
accepted_threshold = 0.7;

% Finding the euclidean distance between the features of both images
%feature_distances = zeros(size(features1,1),size(features2,1));
%for i = 1:size(features2,1)
%    for j = 1:size(features1,1) 
%    feature_distances(j,i) = norm(features2(i,:) - features1(j,:));
%    end
%end


if  strcmp(metric,'HELLINGER')
    
    
%Absolute value - ADDED BY US ORIGINAL PAPER SUGGESTS ONLY
%L1 Normalization of features & Squareroot of features befor
%Euclidean matching
features1 = abs(features1);
features2 = abs(features2);
    

%Hellinger distance As suggested by Arandjelovi ́c, R. and A. 
%Zisserman (2012)- "Three things everyone should know
%to improve object retrieval":

%L1 Normalization of features
features1 = features1/norm(features1,1); 
features2 = features2/norm(features2,1);

%Squareroot of features
features1 = sqrt(features1);
features2 = sqrt(features2);

%Euclidean distance done after these Hellinger modifactions
feature_distances = pdist2(features1,features2, 'euclidean');
    
elseif strcmp(metric,'CHISQUARED')
    %Chi^2 distances
feature_distances  = distChiSq(features1,features2); 

elseif strcmp(metric,'EUCLIDEAN')
    %Euclidean distance
 feature_distances = pdist2(features1, features2, 'EUCLIDEAN');
    
else
 feature_distances = -1;
 assert(feature_distances == -1);
end 


% sort all of the rows in ascending order 
% First column represents nearest neighbors
% Second column represents second nearest neighbors and so on
[sorted_matches, indices] = sort(feature_distances, 2); 


% Calculate the nearest neighbor distance ratio
NNDR = sorted_matches(:,1)./sorted_matches(:,2);

%Filter out nonrelevant features with acceptance threshold
filtered_NNDR = NNDR(NNDR < accepted_threshold);

%Create matched index pairs
indexPairs  = zeros(size(filtered_NNDR,1), 2);

%Get the indices of features 1 from NNDR less than the threshold
indexPairs(:,1) = find(NNDR < accepted_threshold); 

%Get the indices of features 2 from Nearest neighbor sorted distance
%indices within the NNDR threshold
indexPairs(:,2) = indices(NNDR < accepted_threshold, 1);

%sort confidences in descending order (optional step)
[~, ind] = sort(filtered_NNDR, 'descend');
indexPairs  = indexPairs(ind,:);

end