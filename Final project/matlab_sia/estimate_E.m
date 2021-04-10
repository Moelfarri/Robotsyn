function E = estimate_E(xy1, xy2)
    n = size(xy1, 2);
    A = [];
    %the 8 point algorithm
    for i = 1:n
        A_i = [xy2(1,i)*xy1(1,i),xy2(1,i)*xy1(2,i),xy2(1,i), xy2(2,i)*xy1(1,i),xy2(2,i)*xy1(2,i), xy2(2,i), xy1(1,i), xy1(2,i), 1];
        A = [A; A_i];
    end
    [~,~,V] = svd(A);
    smallest_singular = V(:,size(V,2));
    E = reshape(smallest_singular,3,3)';
end