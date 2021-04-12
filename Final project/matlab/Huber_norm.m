function r_sqaured = Huber_norm(r_squared)  
    k = 100;
    r_abs = abs(sqrt(r_squared));
    if r_abs <= k
        r_sqaured =  r_squared;
    else
        r_sqaured = k*(2*r_abs - k);
    end 
end

