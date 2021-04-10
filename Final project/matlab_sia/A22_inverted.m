function A22_inv = A22_inverted(A22)
[row, col] = size(A22);
corres = row/3;
A22_inv = zeros(row, col);
A_temp = zeros(3,3);
for L = 1:corres
    A_temp(:,:) = A22(3*L - 2:3*L,3*L - 2:3*L);
    A22_inv(3*L - 2:3*L,3*L - 2:3*L) = (A_temp)^(-1);
    
end

end