clc;
clear;
format long;

A = [
    0.921, 0, 0.041, 0;
    0, 0.918, 0, 0.033;
    0, 0, 0.924, 0;
    0, 0, 0, 0.937
];
 B = [
    0.017, 0.001;
    0.001, 0.023;
    0, 0.061;
    0.072, 0;
];
C = [
    1,0,0,0; 
    0,1,0,0
];
D = zeros(2, 2);
sys = ss(A, B, C, D, -1);
N = 400;
uSteady = [1; 1];
ySteady = [0.65; 0.77];
L = 30;
n = 4;
m = 2;
p = 2;
coeffR = 10^-4;
coeffQ = 3;
lambdaSigma=1000;
lambdaAlphaEps=0.1;
epsilon=0.005;
R = 10^-4 * eye(2);
Q = 3 * eye(2);
normValue = @(x, matrix) sqrt(x * matrix * x');
l = @(u, y) normValue(u, R)^2 + normValue(y, Q)^2;

objective = @(x) objectiveFunc(x, n, L, l, uSteady, ySteady);

alphaShift = (L + n) * 4;
alphaRange = alphaShift + 1:(L + n) * 4 + N - (L + n) + 1;
sigmaShift = alphaRange(end);
sigmaRange = sigmaShift + 1:(L + n) * 4 + N - (L + n) + 1 + (L + n) * 2;

uData = zeros(1, N * 2);
yData = zeros(1, N * 2);
curX = zeros(1, 4)';

for i = 1:N
    uData(1, [i * 2 - 1 i * 2]) = [ - 1 + 2 / N * (i - 1)];
    %uData(1, [i * 2 - 1 i * 2]) = [-1 + rand * 2 -1 + rand * 2];
    yData(1, [i * 2 - 1 i * 2]) = [(C * curX)];
    curX = A * curX + B * uData(1, [i * 2 - 1 i * 2])';
end

uRes = zeros(N + n + L, 2);
yRes = zeros(N + n + L, 2);
xRes = zeros(N + n + L, 4);

steadyFirstIndex = (L + n) * 4 + (N - (L + n) + 1) + 1;

Aeq = zeros(((L + n) * 4) + (L * 4), ...
            ((L + n) * 4 )+ (N - (L + n) + 1) + (L * 4) + (L + n) *2);
beq = zeros(1, ((L + n) * 4) + (L * 4));
uHankel = hankelMatrix(reshape(uData, 2, N)', L + n, N);
yHankel = hankelMatrix(reshape(yData, 2, N)', L + n, N);

sigmaIndex = 1;
for i = 1:4:(L + n) * 4
    Aeq(i, i) = 1;
    Aeq(i + 1, i + 1) = 1;
    Aeq(i + 2, i + 2) = 1;
    Aeq(i + 3, i + 3) = 1;
    hIndex = fix(i / 4) + 1;
    Aeq(i, alphaRange) = -uHankel(hIndex, :, 1);
    Aeq(i + 1, alphaRange) = -uHankel(hIndex, :, 2);
    Aeq(i + 2, alphaRange) = -yHankel(hIndex, :, 1);
    Aeq(i + 3, alphaRange) = -yHankel(hIndex, :, 2);
    
    Aeq(i + 2, sigmaShift + sigmaIndex) = 1;
    Aeq(i + 3, sigmaShift + sigmaIndex + 1) = 1;
    
    sigmaIndex = sigmaIndex + 2;
end
j = 0;
for i = (L + n) * 4 + 1:numel(Aeq(:, 1))
    Aeq(i, steadyFirstIndex + j) = 1;
    j = j + 1; 
end
beq(1, (L + n) * 4 + 1:(L + n) * 4 + (L * 4)) = 1;

quadH = zeros(((L + n) * 4)  + (N - (L + n) + 1) + (L * 4) + (L + n) * 2);
quadF = zeros(1, ((L + n) * 4)  + (N - (L + n) + 1) + (L * 4) + (L + n) * 2);
for i = alphaShift+1:alphaRange(end)
    quadH(i,i)=lambdaAlphaEps;
end
for i = sigmaShift+1:sigmaRange(end)
    quadH(i,i)=lambdaSigma;
end
for i = n * 4 + 1:4:(L + n) * 4
    quadH(i, i) = coeffR;
    quadH(i + 1, i + 1) = coeffR;
    quadH(i + 2, i + 2) = coeffQ;
    quadH(i + 3, i + 3) = coeffQ;
    for j = steadyFirstIndex:4:numel(quadH(1, :))
        quadH(i, j) = -2 * uSteady(1) * coeffR;
        quadH(i + 1, j + 1) = -2 * uSteady(2) * coeffR;
        quadH(i + 2, j + 2) = -2 * ySteady(1) * coeffQ;
        quadH(i + 3, j + 3) = -2 * ySteady(2) * coeffQ;
    end
end
squares = L * 4 * ...
          (coeffR * uSteady(1)^2 + coeffR * uSteady(2)^2 ...
          + coeffQ * ySteady(1)^2 + coeffQ * ySteady(2)^2);
quadF(1, steadyFirstIndex) = squares;
quadH = quadH * 2;


curX = zeros(1, 4);
disp('MPC loop started.');
for t = 0:n:N
    disp('iteration');
    disp(t);
    
    % Main constraint.
    timeIndex = n + 1 + t;
    Aeq(1:n * 4, alphaRange) = zeros(n * 4, N - (L + n) + 1);
    j = t + 1;
    for i = 1:4:L * 4
        Aeq(i + 2, sigmaRange) = 0;
        Aeq(i + 3, sigmaRange) = 0;
        
        beq(i) = uRes(j, 1);
        beq(i + 1) = uRes(j, 2);
        beq(i + 2) = yRes(j, 1);
        beq(i + 3) = yRes(j, 2);
        j = j + 1;
   end
    
    % Terminal constraint.
    for i = L * 4 + 1:4:(L + n) * 4
        Aeq(i + 2, sigmaRange) = 0;
        Aeq(i + 3, sigmaRange) = 0;
        
        beq(i) = uSteady(1, 1);
        beq(i + 1) = uSteady(2, 1);
        beq(i + 2) = ySteady(1, 1);
        beq(i + 3) = ySteady(2, 1);
    end
    options = optimoptions('quadprog', ...
                           'MaxIter', 100, ....
                           'TolFun', 1e-15, ...
                           'TolX', 1e-15);
    [res, value] = quadprog(quadH, quadF, [], [], ...
                            Aeq, beq, [], [], [], options);    
    res = res(1:(L + n) * 4);
    j = timeIndex - n;
    for i = 1:4:(L + n) * 4
        uRes(j, 1) = res(i);
        uRes(j, 2) = res(i + 1);
        disp(xRes(j,:));
        [dsResY, dsResX] = dynamicSystemFunc(uRes(j,:)', xRes(j,:)', A, B, C);
        yRes(j,:) = dsResY + rand(2,1) * epsilon;
        j = j + 1;
        xRes(j,:) = dsResX';
    end
end
hold off;
hold on;

plt1 = plot(yRes(:, 1));
plt2 = plot(ones(N + 10, 1) * (ySteady(1)));
ylim([0, 1.2]);
xlim([0,N]);

plt3 = plot(yRes(:, 2));
plt4 = plot(ones(N + 10, 1) * (ySteady(2)));
ylim([0, 1.2]);
xlim([0,N]);

plt1.LineWidth = 2;
plt2.LineWidth = 2;
plt3.LineWidth = 2;
plt4.LineWidth = 2;

grid on;

xlabel('MPC iteration');
ylabel('y_1, y_2');
hold off;

