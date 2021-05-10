function [c,ceq] = nonlinconfunc(x)
    %----- Ограничения на sigma_k
    L = 20;
    N = 100;
    n = 4;
    epsilon=0.002;
    alphaShift = (L + n) * 4;
    alphaRange = alphaShift + 1:(L + n) * 4 + N - (L + n) + 1;
    sigmaShift = alphaRange(end);
    sigmaRange = sigmaShift + 1:(L + n) * 4 + N - (L + n) + 1 + (L + n) * 2;
    sigmaValues = x(sigmaRange);
    alphaValues = x(alphaRange);
    
    alphaNorm=0;
    for i=1:numel(alphaValues)
        alphaNorm=alphaNorm+abs(alphaValues(i));
    end
    
    rightPart=epsilon+epsilon*alphaNorm;
    
    retc=[];
    
    for i=1:2:numel(sigmaValues)
        tmp=max(abs(sigmaValues(i)),abs(sigmaValues(i+1)))-rightPart;
        retc=[retc,tmp];
    end
    
    c=retc;
    ceq=[];
end

