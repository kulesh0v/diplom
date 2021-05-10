function [ y, curX ] = dynamicSystemFunc( u, x, A, B, C)
curX = A * x + B * u;
y = C * x;
end

