function retval = hankelMatrix(x, dim, N)
    H = [];
    shift = 0;
    for i = 1:dim
        for j = 1:N - dim + 1
            H(i, j, :) = x(shift + j, :); 
        end
        shift = shift + 1;
    end
    retval = H;
end
