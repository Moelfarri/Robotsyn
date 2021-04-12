function [U, V, W] = JtJ(num_images,J)
    A = J(:,1:6*num_images);
    B = J(:,6*num_images + 1:end);
    U = A.'*A;
    W = A.'*B;
 
    n = size(B,2)/3;
    m = size(B,1)/num_images;
    V = zeros(3*n,3*n);
    for i=1:n
        t = zeros(3,3);
        for j = 1:num_images
            temp = B((j - 1)*m + 1 + (i - 1)*2:(j - 1)*m + 2 + (i - 1)*2,(i - 1)*3 + 1:i*3);
            t = t + temp.'*temp;
        end
        V((i - 1)*3 + 1:i*3,(i - 1)*3 + 1:i*3) = t;
    end
    

      
end

