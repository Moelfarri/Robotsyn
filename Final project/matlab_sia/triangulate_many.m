  
function X = triangulate_many(xy1, xy2, P1, P2)
    % Arguments
    %     xy: Calibrated image coordinates in image 1 and 2
    %         [shape 3 x n]
    %     P:  Projection matrix for image 1 and 2
    %         [shape 3 x 4]
    % Returns
    %     X:  Dehomogenized 3D points in world frame
    %         [shape 4 x n]

    n = size(xy1, 2);
    X = zeros(4, n); 
    A = zeros(4,4);
    for i = 1:n
        A(1,:) = xy1(2,i)*P1(3,:) - P1(2,:);
        A(2,:) = P1(1,:) - xy1(1,i)*P1(3,:);
        A(3,:) = xy2(2,i)*P2(3,:) - P2(2,:);
        A(4,:) = P2(1,:) - xy2(1,i)*P2(3,:);
%         A(2,:) = xy1(2,i)*P1(3,:) - P1(2,:);
%         A(1,:) = xy1(1,i)*P1(3,:) - P1(1,:);
%         A(4,:) = xy2(2,i)*P2(3,:) - P2(2,:);
%         A(3,:) = xy2(1,i)*P2(3,:) - P2(1,:);
        
        [~,~,V] = svd(A);
        X(:,i) = V(:,size(V,2));
        X(:,i) = X(:,i)./X(4,i);
        
    
    end
end