function retval = objectiveFunc(x,n, L, l, uSteady,ySteady)
    uValues = zeros(L, 2);
    yValues = zeros(L, 2);
    j = 1;  
    for i = n * 4 + 1:4:(n + L) * 4
        uValues(j, 1) = x(i);
        uValues(j, 2) = x(i + 1);
        yValues(j, 1) = x(i + 2);
        yValues(j, 2) = x(i + 3);
        j = j + 1;
    end
    retval = sum(arrayfun(@(k) l(uValues(k, :) - uSteady', ...
                 yValues(k, :) - ySteady'), 1:L));
end