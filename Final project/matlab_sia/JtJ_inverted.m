function JTJ_inv = JtJ_inverted(JTJ,num_images)
    A11 = JTJ(1:num_images*6, 1:num_images*6);
    A22 = JTJ(num_images*6 + 1:end, num_images*6 + 1:end);
    
    A12 = JTJ(1:num_images*6, num_images*6 + 1:end);
    A12_T = A12.';
    A22_inv = A22_inverted(A22);
    MD_inv = (A11 - A12*A22_inv*A12_T)^(-1);
    JTJ_inv = [MD_inv, -1*MD_inv*A12*A22_inv;
               -1*A22_inv*A12_T*MD_inv, A22_inv + A22_inv *A12_T * MD_inv * A12 * A22_inv];


end

