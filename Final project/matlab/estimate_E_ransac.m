function [E,best_inliers] = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)

    % Tip: The following snippet extracts a random subset of 8
    % correspondences (w/o replacement) and estimates E using them.
    %   sample = randperm(size(xy1, 2), 8);
    %   E = estimate_E(xy1(:,sample), xy2(:,sample));
    uv1 = K*xy1;
    uv2 = K*xy2;

    highest_inlier_count = 0;
    for i = 1:num_trials
        %randomuly select m = 8 correspondences
        sample = randperm(size(xy1, 2), 8);
        %estimate the essential matrix
        E = estimate_E(xy1(:,sample), xy2(:,sample));
        %calculate the fundamental matrix
        F = F_from_E(E, K);
        %compute the vector of residuals for the full set of
        %correspondences
        e = epipolar_distance(F, uv1, uv2);
        %find all the inliers
        inliers = (abs(e) < distance_threshold);
        %count the number of residuals within the specified threshold
        sum_inlier = sum(inliers(:) == true);
        
        if sum_inlier > highest_inlier_count
            highest_inlier_count = sum_inlier;
            best_inliers = inliers;
        end
    end
  %recalculate the essential matrix for the whole inliers set
  E = estimate_E(xy1(:,best_inliers), xy2(:,best_inliers));
end