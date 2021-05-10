format long


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%THE 2D SYSTEM EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EPS_2d = 0.02;
% [sys_2d, x_init_2d, u_init_2d, y_init_2d, Hu_2d, Hy_2d, GG_2d, gg_2d, umax_2d, umin_2d, u_dim_2d, y_dim_2d, t_y_2d] = generate_the_2d_sys (EPS_2d);
% [isFeasible, u_init_dim2, y_init_dim2, uopt_dim2, yopt_dim2, xopt_dim2, costs_dim2] = ...
%    control_over_T(sys_2d, x_init_2d, u_init_2d, y_init_2d, Hu_2d, Hy_2d, GG_2d, gg_2d, umax_2d, umin_2d, u_dim_2d, y_dim_2d, EPS_2d, t_y_2d);
% plot_the_2d(u_init_dim2, y_init_dim2, uopt_dim2, yopt_dim2, xopt_dim2, costs_dim2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%THE 4D SYSTEM EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EPS_4d = 0.02;
[sys_4d, x_init_4d, u_init_4d, y_init_4d, Hu_4d, Hy_4d, GG_4d, gg_4d, umax_4d, umin_4d, u_dim_4d, y_dim_4d, t_y_4d] = ...
   generate_the_4d_sys(EPS_4d);
[~, u_init_dim4, y_init_dim4, uopt_dim4, yopt_dim4, xopt_dim4, costs_dim4] = ...
    control_over_T(sys_4d, x_init_4d, u_init_4d, y_init_4d, Hu_4d, Hy_4d, GG_4d, gg_4d, umax_4d, umin_4d, u_dim_4d, y_dim_4d, EPS_4d, t_y_4d);
plot_the_4d(u_init_dim4, y_init_dim4, uopt_dim4, yopt_dim4, xopt_dim4, costs_dim4);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% REPEATED OPEN-LOOP OC %%%%%%%%%%%%%%%%%%%%%%%%%
function [isFeasible, u_init, y_init, uopt, yopt, xopt, costs] = control_over_T (sys, x_init, u_init, y_init, Hu, Hy, GG, gg, umax, umin, u_dim, y_dim, EPS, t_y)
    isFeasible = true
    t_init = length(u_init)/u_dim + 1; %when we start control
    T = length(Hu(:, 1))/u_dim; %prediction horizon
    costs = zeros(T-t_init+1,1);
    up = u_init;
    yp = y_init;
    GGtau = GG;
    ggtau = gg;
    for tau = t_init:T   
        Up = Hu(1:(tau-1)*u_dim, :);
        Uf = Hu((tau-1)*u_dim+1:end, :);
        Yp = Hy(1:(tau-1)*y_dim, :);
        Yf = Hy((tau-1)*u_dim+1:end, :);
        
        %-y_dim*2 inequality constraints each time
        %if tau > t_init
        %constraints enforced for the last t_y only

        if tau > T-t_y
            GGtau = GGtau(1+y_dim*2:end, :);
            ggtau = ggtau(1+y_dim*2:end);
        end
        
        if tau > t_init
            GGtau = GGtau(:, 1+y_dim:end);
        end      
        
        [isFeasibleatTau, uf] = control_at_tau (Up, Yp, Uf, Yf, up, yp, EPS, GGtau, ggtau, umax, umin);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %in case control qp is not solved (dunno why, but happened)
        if isFeasibleatTau == false && tau > t_init
            uf = uf_prev(u_dim+1:end);
        end
        uf_prev = uf;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        costs(tau-t_init+1) =  up' * up + uf' * uf; 
        diary on
        tau
        isFeasibleatTau
        uf
        'cost at tau'
        up' * up + uf' * uf 
        diary off
        
        up = vertcat(up, uf(1:u_dim));
        [yf, ~, ~] = lsim(sys, transpose(reshape(up, u_dim, [])), [], x_init);
        yp = vertcat(yp, transpose(yf(end,:)) + EPS * rand([y_dim,1]));  

    end
    uopt = up;
    [yopt, ~, xopt] = lsim(sys, transpose(reshape(uopt, u_dim, [])), [], x_init);
    diary on
    uopt
    yopt
    diary off
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN-LOOP OC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [isFeasible, uf] = control_at_tau ( Up, Yp, Uf, Yf, up, yp, EPS, GG, gg, umax, umin)
   [chiDet, solutionExistsDet] = ...
       estimation_Det (Up, Yp, Uf, Yf, up, yp, EPS, GG);
   %optimal input
   [uf, exitflagDet] = ...
       control_Det (Up, Yp, Uf, Yf, GG, gg, umax, umin, chiDet);

    if exitflagDet == 1 && solutionExistsDet == 1 %both estimation and control problems feasible
        isFeasible = true;
    else
        isFeasible = false;
    end   
end




%%%%%%%%%%%%%%%%%%  ESTIMATION   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [chiDet, allexitflagsEqualOne] = estimation_Det (Up, Yp, Uf, Yf, u_init, y_init, EPS, GG)
%DETERMINANT system: pre-collected data is NOISE-FREE
%Linear Program for ESTIMATION
allexitflagsEqualOne = true;

chiDet = zeros(length(GG(:,1)),1); %for each row of GG   

%precalculate MATRICES for linprog
A = vertcat(Yp, -Yp);
b = vertcat(y_init + EPS.*ones(length(y_init),1), ... 
    -y_init+EPS.*ones(length(y_init),1) ); 
Aeq = vertcat(Up, Uf);
beq = vertcat(u_init, zeros(length(Uf(:,1)), 1));


options = optimoptions('linprog','Display','iter'); %'MaxIter', 15000, ...%'TolCon', 0.001, ...'Preprocess', 'none'
for i = 1:length(GG(:,1))
    [x, fval, exitflag, ~] = ...
        linprog(-GG(i,:)*Yf, ... %-GG(i,:) instead of GG(i, :) for a max instead of min
            A, b, Aeq, beq, ...
                [], [], options);
    
    
    if exitflag == 1
        chiDet(i) = fval;
    else
        allexitflagsEqualOne = false;
        diary on
        'broken linprog'
        exitflag
        diary off
    end

end
chiDet = -chiDet; % minus before chi to contain a max instead of min 
 
end




%%%%%%%%%%%%%%%%%%%%% CONTROL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [uOptDet, exitflag] = control_Det (Up, Yp, Uf, Yf, GG, gg, umax, umin, chiDet) 
%Quadratic Problem for CONTROL
[alphaCdet,~, exitflag, ~] = quadprog(transpose(Uf)*Uf, [], vertcat(GG*Yf, Uf, -Uf), ... %%for -umax< = Uf*alphaCdet <= umax
    vertcat(gg-chiDet, umax * ones(length(Uf(:,1)), 1), -umin * ones(length(Uf(:, 1)), 1)),...
    vertcat(Up, Yp), zeros(size(Up,1)+size(Yp,1),1));

%optimal input
uOptDet = Uf*alphaCdet;
 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%% GENERATION OF A SYSTEM WITH INITIAL FEASIBILITY  %%%%%%%%%%%%%%%

%%%%CUSTOM HANKEL MATRICES BUILDER
function [Hw] = hankel_multidim (w_flatten, dim_w, L)
%L instances in a column
%N instances in the w_flatten
%=> N-L+1 columns
N = size(w_flatten)/dim_w;
Hw = [];
for i=1:N-L+1
    Hw = horzcat(Hw, w_flatten(1+(i-1)*dim_w:(i-1+L)*dim_w));
end
end




%%%%%%%THE 4D SYSTEM
function [sys, x_init, u_init_flatten, y_init_noisy, Hu, Hy, GG, gg, umax, umin, u_dim, y_dim, t_y] = generate_the_4d_sys (EPS)
for attempt = 1:2
    %%%%%
    %%%LTI sys
    %%%%%%%%
    A_SYS = [0.921 0 0.041 0; 0 0.918 0 0.033; 0 0 0.924 0; 0 0 0 0.937];
    B_SYS = [0.017 0.001; 0.001 0.023; 0 0.061; 0.072 0];
    C_SYS = [1 0 0 0; 0 1 0 0]; 
    D_SYS = [0 0; 0 0];
    X0_SYS = [0;0;0;0]; % initial condition

    
%     diary on
%     'observability'
%     rank(C_SYS)
%     rank(vertcat(C_SYS, C_SYS*A_SYS))
%     rank(vertcat(C_SYS, C_SYS*A_SYS, C_SYS*A_SYS*A_SYS))
%     rank(vertcat(C_SYS, C_SYS*A_SYS, C_SYS*A_SYS*A_SYS*A_SYS))
%     diary off
    
    sys = ss(A_SYS, B_SYS, C_SYS, D_SYS, -1);
    
    u_dim=2;
    y_dim=2;
    n = 4; %system dimension
    
    %%%%%%%%%%%%%%
    %Input and output constraints
    %%%%%%%%%%%%%%
    % -y_scale <= y <=y_scale
    y_scale = 0.1;
    % -u_scale <= u <= u_scale
    u_scale = 0.8;
    umax = 1*u_scale;
    umin = -1*u_scale;

    %%%%%%%%%%
    %DATA GENERATION
    %%%%%%%%%%
    L = 55; %prediction horizon
    t_y=20; % we enforce constraints on the last t_y outputs y only
    x_init = [4; 0; 1; -1]; 
    
    %generating an INPUT signal u, uniformly distributed in (-u_scale,u_scale)
    u = -u_scale + 2*u_scale*rand((n+L)*(u_dim+1)-1,u_dim);
    u_flatten = transpose(reshape(u.',1,[]));
    
    %creating a Hankel matrix (n + L) x (n + L + 1)of order n+L for u 
    H_u_check = hankel(u_flatten(1:(n+L)*u_dim), u_flatten((n+L)*u_dim:end));
    %persistency of excitation check 
    rank(H_u_check)
    %hooray, full row rank!!

    %generate OUTPUTS y
    [y, ~, ~] = lsim(sys, u, 0:1:length(u)-1, X0_SYS);
    %Hankel matrix for y, u
    y_flatten = transpose(reshape(y.',1,[]));
    
    Hu = hankel_multidim(u_flatten, u_dim, L);
    Hy = hankel_multidim(y_flatten, y_dim, L);


    %we start control at tau >= n, part of trajectory before n is  generated here
    tau = n;%current position in time

    u_init = u_scale*rand(tau,u_dim); 
    u_init_flatten = transpose(reshape(u_init.',1,[]));
    [y_init, ~, ~] = lsim(sys, u_init, [], x_init);  
    
    %add some noise to y_init
    y_init_flatten = transpose(reshape(y_init.',1,[]));
    y_init_clear = y_init_flatten;
    y_init_noisy = y_init_clear + EPS * rand(length(y_init_flatten), 1);


    %%%%%%%%%%%
    %%%MATRICES at time TAU
    %%%%%%%%%%%
    %Up, Yp, Uf, Yf
    Up = Hu(1:tau*u_dim, :);
    Uf = Hu(tau*u_dim+1:end, :);
    Yp = Hy(1:tau*y_dim, :);
    Yf = Hy(tau*y_dim+1:end, :);

    %GG & gg - block matrices formed from G(t), g(t) 
    gg = ones(2*(L-tau)*y_dim,1)*y_scale;
    GG = zeros(2*(L -tau)*y_dim, (L-tau)*y_dim);
    for i = 1:(L-tau)*y_dim
        GG(2*i-1,i) = 1;
        GG(2*i,i) = -1; 
    end

    gg = gg(end-2*t_y*y_dim+1:end);
    GG = GG(end-2*t_y*y_dim+1:end, :);
    
    %initial estimation
    [chiDet, solutionExistsDet] = ...
        estimation_Det (Up, Yp, Uf, Yf, u_init_flatten, y_init_noisy, EPS, GG);
    
    %initial control
    [uOptDet, exitflagDet] = ...
        control_Det (Up, Yp, Uf, Yf, GG, gg, umax, umin, chiDet);
    diary on 
    solutionExistsDet
    exitflagDet
    diary off

    if exitflagDet == 1 && solutionExistsDet == 1 %initial feasibility granted
        break;
    end

end %end of attempts to generate
diary on
attempt
diary off
end



%THE 2D SYS
function [sys, x_init, u_init_flatten, y_init_noisy, Hu, Hy, GG, gg, umax, umin, u_dim, y_dim, t_y] = generate_the_2d_sys (EPS)
for attempt = 1:10
    %%%%%
    %%%LTI sys
    %%%%%%%%
    A_SYS = [0.9950 0.0998; -0.0998 0.9950];
    B_SYS = [0.0050; 0.0998];
    C_SYS = [1 0]; 
    D_SYS = [0];
    X0_SYS = [0; 0]; % initial condition
    u_dim=1;
    y_dim=1;
    sys = ss(A_SYS, B_SYS, C_SYS, D_SYS, -1);
    
    %%%%%%%%%%%%%%
    %Input and output constraints
    %%%%%%%%%%%%%%
    % -1 <= y <=1
    G = [1; -1];
    g = [1;1]; 
    % -1 <= u <= 1
    u_scale = 0.7;
    umax = 1*u_scale;
    umin = -1*u_scale;
    y_scale= 0.3;
    
    %enforce constraints on the last t_y outputs
    t_y=15;

    %%%%%%%%%%
    %DATA GENERATION
    %%%%%%%%%% 
    n = 2; %system dimension
    L = 135; 
    x_init = [5; -2]; 
    
    %generating an INPUT signal u, uniformly distributed in (0,u_scale)
    u = u_scale*rand((n+L)*(u_dim+1)-1,u_dim);
    u_flatten = transpose(reshape(u.',1,[]));
    
    %creating a Hankel matrix (n + L) x (n + L + 1)of order n+L for u 
    H_u_check = hankel(u_flatten(1:(n+L)*u_dim), u_flatten((n+L)*u_dim:end));
    %persistency of excitation check
    rank(H_u_check)

    %generate OUTPUTS y
    [y, ~, ~] = lsim(sys, u, 0:1:length(u)-1, X0_SYS);
    %Hankel matrix for y, u
    y_flatten = transpose(reshape(y.',1,[]));
    Hy = hankel(y_flatten(1:L*y_dim), y_flatten(L*y_dim:end));
    Hu = hankel(u_flatten(1:L*u_dim), u_flatten(L*u_dim:end));

   
    tau = n;
    u_init = [0; 0]; 
    u_init_flatten = transpose(reshape(u_init.',1,[]));
    [y_init, ~, ~] = lsim(sys, u_init, [], x_init);

    %add some noise to y_init
    y_init_flatten = transpose(reshape(y_init.',1,[]));
    y_init_clear = y_init_flatten;
    y_init_noisy = y_init_clear + EPS * rand(length(y_init_flatten), 1);


    %%%%%%%%%%%
    %%%MATRICES at time TAU
    %%%%%%%%%%%
    %Up, Yp, Uf, Yf
    Up = Hu(1:tau*u_dim, :);
    Uf = Hu(tau*u_dim+1:end, :);
    Yp = Hy(1:tau*y_dim, :);
    Yf = Hy(tau*y_dim+1:end, :);

    gg = ones(2*(L-tau)*y_dim,1)*y_scale;
    GG = zeros(2*(L-tau)*y_dim, (L-tau)*y_dim);
    for i = 1:(L-tau)*y_dim
        GG(2*i-1,i) = 1;
        GG(2*i,i) = -1; 
    end
    

    gg = gg(end-2*t_y*y_dim+1:end);
    GG = GG(end-2*t_y*y_dim+1:end, :);

    [chiDet, solutionExistsDet] = ...
        estimation_Det (Up, Yp, Uf, Yf, u_init_flatten, y_init_noisy, EPS, GG);
    
    [uOptDet, exitflagDet] = ...
        control_Det (Up, Yp, Uf, Yf, GG, gg, umax, umin, chiDet);
    
    
    diary on 
    exitflagDet
    solutionExistsDet
    chiDet
    diary off

    if exitflagDet == 1 && solutionExistsDet == 1 %Initial feasibility granted
        break;
    end

end %end of attempts to generate
diary on
attempt
diary off
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_the_2d (u_init, y_init, uopt, yopt, xopt, costs)
    umax = 0.7;
    ymax = 0.3;
    figure
    stairs(0:length(uopt)-1, uopt, 'k', 'LineWidth', 1.7);
    axis([0 (length(uopt)-1) -1 1])
    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$u$', 'Interpreter', 'latex')
    hold on 
    yline(umax, '-.');
    yline(-umax, '-.');
    hold off
    
    figure
    plot(0:length(yopt)-1, yopt, 'k', 'LineWidth', 1.7);
    axis tight
    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$y$', 'Interpreter', 'latex')
    hold on 
    yline(ymax, '-.');
    yline(-ymax, '-.');
    hold off
    
    figure
    x1 = xopt(:,1);
    x2 = xopt(:,2);
    plot(x1, x2, 'k', 'LineWidth', 1.7)
    xlabel('$x_1$', 'Interpreter', 'latex')
    ylabel('$x_2$', 'Interpreter', 'latex')
    
    figure
    plot(2:2+29-1, costs(1:29), 'k', 'LineWidth', 1.7);
    xlim([0 (29-1)])
    xlabel('$\tau$', 'Interpreter', 'latex')
    ylabel('$J(\tau)$', 'Interpreter', 'latex')
       
end


%helper fcn for plot_the_4d
function [w1, w2] = vector_into_two(w)
%length of w is assumed to be even
w_two_rows = reshape(w, 2, []);
w1 = transpose(w_two_rows(1, :));
w2 = transpose(w_two_rows(2, :));
end

function plot_the_4d (u_init, y_init, uopt, yopt, xopt, costs)
    umax = 0.8;
    ymax = 0.1;
    
    [uopt_1, uopt_2] = vector_into_two(uopt);
    yopt_1 = yopt(:, 1);
    yopt_2 = yopt(:, 2);
    
    diary on
    yopt_1
    yopt_2
    diary off
    
    figure
    stairs(0:length(uopt_1)-1, uopt_1, 'k', 'LineWidth', 1.7);
    axis([0 (length(uopt_1)-1) -1 1])
    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$u_1$', 'Interpreter', 'latex')
    hold on 
    yline(umax, '-.');
    yline(-umax, '-.');
    hold off
    
    figure
    stairs(0:length(uopt_2)-1, uopt_2, 'k', 'LineWidth', 1.7);
    axis([0 (length(uopt_1)-1) -1 1])
    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$u_2$', 'Interpreter', 'latex')
    hold on 
    yline(umax, '-.');
    yline(-umax, '-.');
    hold off
    
    figure
    plot(0:length(yopt_1)-1, yopt_1, '--k', 'LineWidth', 1.5);
    axis tight
    xlabel('$t$', 'Interpreter', 'latex')
    ylabel('$y$', 'Interpreter', 'latex')
    hold on 
    plot(0:length(yopt_2)-1, yopt_2, '-k', 'LineWidth', 1.5);
    yline(ymax, '-.');
    yline(-ymax, '-.');
    hold off
    legend('$y_1$', '$y_2$', 'Interpreter', 'latex')
    
    
    figure
    x3 = xopt(:,3);
    x4 = xopt(:,4);
    plot(x3, x4, 'k', 'LineWidth', 2);   
    xlabel('x3')
    ylabel('x4')
    
    figure
    plot(4:4+length(costs)-1, costs, 'k', 'LineWidth', 1.7);
    xlim([0 (length(costs)-1)])
    xlabel('$\tau$', 'Interpreter', 'latex')
    ylabel('$J(\tau)$', 'Interpreter', 'latex')
    
end