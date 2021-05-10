function retval = objectiveFunc(x,n, L, l, uSteady,ySteady, sigmaRange, alphaRange, lambdaAlphaEps, lambdaSigma)
    ret = 0;
    uValues = zeros(L, 2);
    yValues = zeros(L, 2);
    
    % добавляем нормы векторов альфа и сигма
    %-----------------------------------------
    sigmaValues = x(sigmaRange);
    alphaValues = x(alphaRange);
    
    alphaSum=0;
    for i=1:numel(alphaValues)
        alphaSum=alphaSum+alphaValues(i)^2;
    end
    alphaSum=alphaSum*lambdaAlphaEps;
    
    ret=ret+alphaSum;
    
    sigmaSum=0;
    for i=1:numel(sigmaValues)
        sigmaSum=sigmaSum+sigmaValues(i)^2;
    end
    sigmaSum=sigmaSum*lambdaSigma;
    
    ret=ret+sigmaSum;
    %------------------------------------
    
    j = 1;  
    for i = n * 4 + 1:4:(n + L) * 4
        uValues(j, 1) = x(i);
        uValues(j, 2) = x(i + 1);
        yValues(j, 1) = x(i + 2);
        yValues(j, 2) = x(i + 3);
        j = j + 1;
    end
    ret = ret + sum(arrayfun(@(k) l(uValues(k, :) - uSteady', ...
                 yValues(k, :) - ySteady'), 1:L));
    retval = ret;
end