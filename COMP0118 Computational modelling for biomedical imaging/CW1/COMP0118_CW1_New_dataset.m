clear;
close all;

%% Importing the data

% Load the diffusion signal
fid = fopen('Archive/isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); % Read in the header
D = fscanf(fid, '%f', [6, inf])'; % Read in the data
fclose(fid);

% Select the first of the 6 voxels
meas = D(:,1);

% Load the protocol
fid = fopen('Archive/isbi2015_protocol.txt', 'r', 'b');
fgetl(fid);
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);

% Create the protocol
grad_dirs = A(1:3,:);
G = A(4,:);
delta = A(5,:);
smalldel = A(6,:);
TE = A(7,:);
GAMMA = 2.675987E8;
bvals = ((GAMMA*smalldel.*G).^2).*(delta-smalldel/3);

% convert bvals units from s/m^2 to s/mm^2
bvals = bvals/10^6;

% slides L03
qhat = GAMMA * smalldel .* G;


%% 1.3.1 We will start by identifying the RESNORM values, and some starting points
% USING FMINCON

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=BallStickSSDOptim4(startx, meas, bvals, grad_dirs, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_bs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_bs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_bs(1);
diff = x_con_bs(2);
f = x_con_bs(3);
theta = x_con_bs(4);
phi = x_con_bs(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
% Estimated model
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=BallStickSSDOptim3(startx, meas, bvals, grad_dirs, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_bs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_bs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_bs(1)^2;
diff = x_unc_bs(2)^2;
f = exp(-(x_unc_bs(3)^2));
theta = (exp(-(x_unc_bs(4)^2)))*pi;
phi = (exp(-(x_unc_bs(5)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
str2 = 'x1='+string(x_unc_bs(1))+'; x2='+string(x_unc_bs(2))+'; x3='+string(x_unc_bs(3))+'; x4='+string(x_unc_bs(4))+'; x5='+string(x_unc_bs(5));
xlabel({str1, str2});


%% 1.3.1 Let's estimate a starting value

% Desgin matrix for linear estimation
Gdesign = [ones(1, length(bvals)); -bvals].';
x = pinv(Gdesign)*log(meas); 
S0_lin_b = exp(x(1));
d_lin_b = x(2);
S0_lin = max(0.1, S0_lin_b);
d_lin = max(1e-7, d_lin_b);


%% 1.3.1 let's run the plot again with the informed point

% USING FMINCON

% Parameters 
nb_of_retries = 5;
nb_of_steps = 20;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',15000,'Algorithm','interior-point','TolX',1e-7,'TolFun',1e-7);

% Informed point
S0_lin = max(0.1, S0_lin);
d_lin = max(0, d_lin);
startx_0 = [S0_lin d_lin 5e-01 1 1];
% Previous point
%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [est_params,RESNORM,~,~]=BallStickSSDOptim4(startx, meas, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
        if(kk==1 && ll==1)
            best_resnorm = RESNORM;
            best_params = est_params;
        else
            if(RESNORM<best_resnorm)
                best_resnorm = RESNORM;
                best_params = est_params;
            end
        end
    end
end

figure();
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');

disp('Best parameters from the informed point: '+string(best_params));


%% Computing lambdas for the Zeppelin models

logMeas = log(meas); %should be defined...

% Let's construct the design matrix
G = zeros(3612, 7); % We have 108 data points (rows) and 7 unknowns (columns)
G(:, 1) = ones(3612, 1);
G(:, 2) = (-bvals .* (grad_dirs(1, :).^2)).';
G(:, 3) = (-2*bvals .* (grad_dirs(1, :).*grad_dirs(2, :))).';
G(:, 4) = (-2*bvals .* (grad_dirs(1, :).*grad_dirs(3, :))).';
G(:, 5) = (-bvals .* (grad_dirs(2, :).^2)).';
G(:, 6) = (-2*bvals .* (grad_dirs(2, :).*grad_dirs(3, :))).';
G(:, 7) = (-bvals .* (grad_dirs(3, :).^2)).';

% Computing the x results vector
x_res = pinv(G) * logMeas;

% Recovering the elements of S(0,0) and D from x_res
logMeas0 = x_res(1);
D_res = [[x_res(2), x_res(3), x_res(4)]; [x_res(3), x_res(5), x_res(6)]; [x_res(4), x_res(6), x_res(7)]];

e = eig(D_res);

lambda1 = max(e);
lambda2 = min(e);


%% Zeppelin model

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=ZeppelinStickSSDOptim4(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_zs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_zs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_zs(1);
diff = x_con_zs(2);
f = x_con_zs(3);
theta = x_con_zs(4);
phi = x_con_zs(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
% Estimated model
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=ZeppelinStickSSDOptim3(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_zs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_zs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_zs(1)^2;
diff = x_unc_zs(2)^2;
f = exp(-(x_unc_zs(3)^2));
theta = (exp(-(x_unc_zs(4)^2)))*pi;
phi = (exp(-(x_unc_zs(5)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
str2 = 'x1='+string(x_unc_zs(1))+'; x2='+string(x_unc_zs(2))+'; x3='+string(x_unc_zs(3))+'; x4='+string(x_unc_zs(4))+'; x5='+string(x_unc_zs(5));
xlabel({str1, str2});


%% TORTUOSITY model

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim4(startx, meas, bvals, grad_dirs, lambda1, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_zst = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_zst = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_zst(1);
diff = x_con_zst(2);
f = x_con_zst(3);
theta = x_con_zst(4);
phi = x_con_zst(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
% Estimated model
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim3(startx, meas, bvals, grad_dirs, lambda1, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_zst = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_zst = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_zst(1)^2;
diff = x_unc_zst(2)^2;
f = exp(-(x_unc_zst(3)^2));
theta = (exp(-(x_unc_zst(4)^2)))*pi;
phi = (exp(-(x_unc_zst(5)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
str2 = 'x1='+string(x_unc_zst(1))+'; x2='+string(x_unc_zst(2))+'; x3='+string(x_unc_zst(3))+'; x4='+string(x_unc_zst(4))+'; x5='+string(x_unc_zst(5));
xlabel({str1, str2});


%% AIC for the DT model

S_dt = exp(logMeas0)*exp(-bvals*grad_dirs.'*D_res*grad_dirs);

plot(meas, 'b+');
hold on;
plot(S_dt, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'DT model','SSD = '+string(sum((meas-S_dt.').^2))});

N = 2;
K = length(bvals);

AIC_DT = 2*N + K*log((1/K)*sum((meas-S_dt.').^2));
BIC_DT = N*log(K) + K*log((1/K)*sum((meas-S_dt.').^2));

%% Computing AIC for the three models

N = 5;
K = length(bvals);

AIC_con_bs = 2*N + K*log((1/K)*BallStickSSD(x_con_bs, meas, bvals, grad_dirs));
AIC_unc_bs = 2*N + K*log((1/K)*BallStickSSDUNC(x_unc_bs, meas, bvals, grad_dirs));

AIC_con_zs = 2*N + K*log((1/K)*ZeppelinStickSSD(x_con_zs, meas, bvals, grad_dirs, lambda1, lambda2));
AIC_unc_zs = 2*N + K*log((1/K)*ZeppelinStickSSDUNC(x_unc_zs, meas, bvals, grad_dirs, lambda1, lambda2));

AIC_con_zst = 2*N + K*log((1/K)*ZeppelinStickTortuositySSD(x_con_zst, meas, bvals, grad_dirs, lambda1));
AIC_unc_zst = 2*N + K*log((1/K)*ZeppelinStickTortuositySSDUNC(x_unc_zst, meas, bvals, grad_dirs, lambda1));

AIC_bs = min(AIC_con_bs, AIC_unc_bs);
AIC_zs = min(AIC_con_zs, AIC_unc_zs);
AIC_zst = min(AIC_con_zst, AIC_unc_zst);


%% Computing BIC for the three models

BIC_con_bs = N*log(K) + K*log((1/K)*BallStickSSD(x_con_bs, meas, bvals, grad_dirs));
BIC_unc_bs = N*log(K) + K*log((1/K)*BallStickSSDUNC(x_unc_bs, meas, bvals, grad_dirs));

BIC_con_zs = N*log(K) + K*log((1/K)*ZeppelinStickSSD(x_con_zs, meas, bvals, grad_dirs, lambda1, lambda2));
BIC_unc_zs = N*log(K) + K*log((1/K)*ZeppelinStickSSDUNC(x_unc_zs, meas, bvals, grad_dirs, lambda1, lambda2));

BIC_con_zst = N*log(K) + K*log((1/K)*ZeppelinStickTortuositySSD(x_con_zst, meas, bvals, grad_dirs, lambda1));
BIC_unc_zst = N*log(K) + K*log((1/K)*ZeppelinStickTortuositySSDUNC(x_unc_zst, meas, bvals, grad_dirs, lambda1));

BIC_bs = min(BIC_con_bs, BIC_unc_bs);
BIC_zs = min(BIC_con_zs, BIC_unc_zs);
BIC_zst = min(BIC_con_zst, BIC_unc_zst);

%% Ranking AIC/BIC

nb_of_retries = 3;
nb_of_steps = 12;

AIC_bs_res = zeros(6, 1);
AIC_zs_res = zeros(6, 1);
AIC_zst_res = zeros(6, 1);
BIC_bs_res = zeros(6, 1);
BIC_zs_res = zeros(6, 1);
BIC_zst_res = zeros(6, 1);
AIC_2bs_res = zeros(6, 1);
AIC_2zs_res = zeros(6, 1);
AIC_2zst_res = zeros(6, 1);
BIC_2bs_res = zeros(6, 1);
BIC_2zs_res = zeros(6, 1);
BIC_2zst_res = zeros(6, 1);

tic;
for bb=1:6
    meas = D(:, bb);

    % Desgin matrix for linear estimation
    Gdesign = [ones(1, length(bvals)); -bvals].';
    x = pinv(Gdesign)*log(meas); 
    S0_lin_b = exp(x(1));
    d_lin_b = x(2);
    S0_lin = max(0.3, S0_lin_b);
    d_lin = max(1e-7, d_lin_b);
    d_lin = min(2, d_lin);

    % Zeppelin parameter estimation
    logMeas = log(meas);
    x_res = pinv(G) * logMeas;
    logMeas0 = x_res(1);
    D_res = [[x_res(2), x_res(3), x_res(4)]; [x_res(3), x_res(5), x_res(6)]; [x_res(4), x_res(6), x_res(7)]];
    e = eig(D_res);
    lambda1 = max(e);
    lambda2 = min(e);

    if(isnan(lambda1))
        lambda1 = 0.5;
        lambda2 = 0.01;
    end

    % Ball and stick

    % Parameters 
    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0];
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=BallStickSSDOptim4(startx, meas, bvals, grad_dirs, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_con_bs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_con_bs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_bs(1);
    diff = x_con_bs(2);
    f = x_con_bs(3);
    theta = x_con_bs(4);
    phi = x_con_bs(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    % Estimated model
    S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];
    
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=BallStickSSDOptim3(startx, meas, bvals, grad_dirs, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_unc_bs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_unc_bs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Extract the parameters
    S0 = x_unc_bs(1)^2;
    diff = x_unc_bs(2)^2;
    f = exp(-(x_unc_bs(3)^2));
    theta = (exp(-(x_unc_bs(4)^2)))*pi;
    phi = (exp(-(x_unc_bs(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    % Zeppelin and stick
    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0];
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=ZeppelinStickSSDOptim4(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_con_zs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_con_zs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_zs(1);
    diff = x_con_zs(2);
    f = x_con_zs(3);
    theta = x_con_zs(4);
    phi = x_con_zs(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    % Estimated model
    S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];
    
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=ZeppelinStickSSDOptim3(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_unc_zs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_unc_zs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Extract the parameters
    S0 = x_unc_zs(1)^2;
    diff = x_unc_zs(2)^2;
    f = exp(-(x_unc_zs(3)^2));
    theta = (exp(-(x_unc_zs(4)^2)))*pi;
    phi = (exp(-(x_unc_zs(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    % Zeppelin and stick with tortuosity
    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0];
    
    if(bb~=5)
        for ll=1:nb_of_retries
            for kk=1:nb_of_steps
                startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [x,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim4(startx, meas, bvals, grad_dirs, lambda1, h);
                res_plot(ll, kk) = RESNORM;
                if(ll==1 && kk==1)
                    x_con_zst = x;
                    best_resnorm = RESNORM;
                end
                if(RESNORM<best_resnorm)
                    x_con_zst = x;
                    best_resnorm = RESNORM;
                end
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_zst(1);
    diff = x_con_zst(2);
    f = x_con_zst(3);
    theta = x_con_zst(4);
    phi = x_con_zst(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    % Estimated model
    S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
  
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0];
    
    if(bb~=5)
        for ll=1:nb_of_retries
            for kk=1:nb_of_steps
                startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [x,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim3(startx, meas, bvals, grad_dirs, lambda1, h);
                res_plot(ll, kk) = RESNORM;
                if(ll==1 && kk==1)
                    x_unc_zst = x;
                    best_resnorm = RESNORM;
                end
                if(RESNORM<best_resnorm)
                    x_unc_zst = x;
                    best_resnorm = RESNORM;
                end
            end
        end
    end
    
    % Extract the parameters
    S0 = x_unc_zst(1)^2;
    diff = x_unc_zst(2)^2;
    f = exp(-(x_unc_zst(3)^2));
    theta = (exp(-(x_unc_zst(4)^2)))*pi;
    phi = (exp(-(x_unc_zst(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(grad_dirs.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

    % Ball and 2 sticks

    % Parameters 
    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=TwoBallStickSSDOptim4(startx, meas, bvals, grad_dirs, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_con_2bs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_con_2bs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_2bs(1);
    diff = x_con_2bs(2);
    f1 = x_con_2bs(3);
    theta1 = x_con_2bs(4);
    phi1 = x_con_2bs(5);
    f2 = x_con_2bs(6);
    theta2 = x_con_2bs(7);
    phi2 = x_con_2bs(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];
    
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=TwoBallStickSSDOptim3(startx, meas, bvals, grad_dirs, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_unc_2bs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_unc_2bs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_unc_2bs(1);
    diff = x_unc_2bs(2);
    f1 = x_unc_2bs(3);
    theta1 = x_unc_2bs(4);
    phi1 = x_unc_2bs(5);
    f2 = x_unc_2bs(6);
    theta2 = x_unc_2bs(7);
    phi2 = x_unc_2bs(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    
    % Zeppelin and two sticks
    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=TwoZeppelinStickSSDOptim4(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_con_2zs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_con_2zs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_2zs(1);
    diff = x_con_2zs(2);
    f1 = x_con_2zs(3);
    theta1 = x_con_2zs(4);
    phi1 = x_con_2zs(5);
    f2 = x_con_2zs(6);
    theta2 = x_con_2zs(7);
    phi2 = x_con_2zs(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
    S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];
    
    
    for ll=1:nb_of_retries
        for kk=1:nb_of_steps
            startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
            [x,RESNORM,~,~]=TwoZeppelinStickSSDOptim3(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
            res_plot(ll, kk) = RESNORM;
            if(ll==1 && kk==1)
                x_unc_2zs = x;
                best_resnorm = RESNORM;
            end
            if(RESNORM<best_resnorm)
                x_unc_2zs = x;
                best_resnorm = RESNORM;
            end
        end
    end
    
    % Extract the parameters
    % Recovering the parameters from the output of fminunc
    S0 = x_unc_2zs(1);
    diff = x_unc_2zs(2);
    f1 = x_unc_2zs(3);
    theta1 = x_unc_2zs(4);
    phi1 = x_unc_2zs(5);
    f2 = x_unc_2zs(6);
    theta2 = x_unc_2zs(7);
    phi2 = x_unc_2zs(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
    S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    
    % Zeppelin and two sticks with tortuosity

    nb_of_retries = 5;
    nb_of_steps = 23;
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
    startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];
    
    if(bb~=5)
        for ll=1:nb_of_retries
            for kk=1:nb_of_steps
                startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [x,RESNORM,~,~]=TwoZeppelinStickTortuositySSDOptim4(startx, meas, bvals, grad_dirs, lambda1, h);
                res_plot(ll, kk) = RESNORM;
                if(ll==1 && kk==1)
                    x_con_2zst = x;
                    best_resnorm = RESNORM;
                end
                if(RESNORM<best_resnorm)
                    x_con_2zst = x;
                    best_resnorm = RESNORM;
                end
            end
        end
    end
    
    % Recovering the parameters from the output of fminunc
    S0 = x_con_2zst(1);
    diff = x_con_2zst(2);
    f1 = x_con_2zst(3);
    theta1 = x_con_2zst(4);
    phi1 = x_con_2zst(5);
    f2 = x_con_2zst(6);
    theta2 = x_con_2zst(7);
    phi2 = x_con_2zst(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
    S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    
    res_plot = zeros(nb_of_retries, nb_of_steps);
    h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8,'Display','off');
    
    %startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
    startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];
    
    if(bb~=5)
        for ll=1:nb_of_retries
            for kk=1:nb_of_steps
                startx = startx_0 + [normrnd(0, 3) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [x,RESNORM,~,~]=TwoZeppelinStickTortuositySSDOptim3(startx, meas, bvals, grad_dirs, lambda1, h);
                res_plot(ll, kk) = RESNORM;
                if(ll==1 && kk==1)
                    x_unc_2zst = x;
                    best_resnorm = RESNORM;
                end
                if(RESNORM<best_resnorm)
                    x_unc_2zst = x;
                    best_resnorm = RESNORM;
                end
            end
        end
    end
    
    % Extract the parameters
    % Recovering the parameters from the output of fminunc
    S0 = x_unc_2zst(1);
    diff = x_unc_2zst(2);
    f1 = x_unc_2zst(3);
    theta1 = x_unc_2zst(4);
    phi1 = x_unc_2zst(5);
    f2 = x_unc_2zst(6);
    theta2 = x_unc_2zst(7);
    phi2 = x_unc_2zst(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
    fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
    fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
    fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
    fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
    S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));


    N = 5;
    K = length(bvals);
    
    AIC_con_bs = 2*N + K*log((1/K)*BallStickSSD(x_con_bs, meas, bvals, grad_dirs));
    AIC_unc_bs = 2*N + K*log((1/K)*BallStickSSDUNC(x_unc_bs, meas, bvals, grad_dirs));
    
    AIC_con_zs = 2*N + K*log((1/K)*ZeppelinStickSSD(x_con_zs, meas, bvals, grad_dirs, lambda1, lambda2));
    AIC_unc_zs = 2*N + K*log((1/K)*ZeppelinStickSSDUNC(x_unc_zs, meas, bvals, grad_dirs, lambda1, lambda2));
    
    if(bb==5)
        AIC_con_zst = -1e4;
        AIC_unc_zst = -1e4;
    else
        AIC_con_zst = 2*N + K*log((1/K)*ZeppelinStickTortuositySSD(x_con_zst, meas, bvals, grad_dirs, lambda1));
        AIC_unc_zst = 2*N + K*log((1/K)*ZeppelinStickTortuositySSDUNC(x_unc_zst, meas, bvals, grad_dirs, lambda1));
    end

    AIC_bs = min(AIC_con_bs, AIC_unc_bs)
    AIC_zs = min(AIC_con_zs, AIC_unc_zs)
    AIC_zst = min(AIC_con_zst, AIC_unc_zst)
    
    BIC_con_bs = N*log(K) + K*log((1/K)*BallStickSSD(x_con_bs, meas, bvals, grad_dirs));
    BIC_unc_bs = N*log(K) + K*log((1/K)*BallStickSSDUNC(x_unc_bs, meas, bvals, grad_dirs));
    
    BIC_con_zs = N*log(K) + K*log((1/K)*ZeppelinStickSSD(x_con_zs, meas, bvals, grad_dirs, lambda1, lambda2));
    BIC_unc_zs = N*log(K) + K*log((1/K)*ZeppelinStickSSDUNC(x_unc_zs, meas, bvals, grad_dirs, lambda1, lambda2));
    
    if(bb==5)
        BIC_con_zst = -1e4;
        BIC_unc_zst = -1e4;
    else
        BIC_con_zst = N*log(K) + K*log((1/K)*ZeppelinStickTortuositySSD(x_con_zst, meas, bvals, grad_dirs, lambda1));
        BIC_unc_zst = N*log(K) + K*log((1/K)*ZeppelinStickTortuositySSDUNC(x_unc_zst, meas, bvals, grad_dirs, lambda1));
    end
    
    BIC_bs = min(BIC_con_bs, BIC_unc_bs)
    BIC_zs = min(BIC_con_zs, BIC_unc_zs)
    BIC_zst = min(BIC_con_zst, BIC_unc_zst)

    N = 8;

    AIC_con_2bs = 2*N + K*log((1/K)*TwoBallStickSSD(x_con_2bs, meas, bvals, grad_dirs));
    AIC_unc_2bs = 2*N + K*log((1/K)*TwoBallStickSSDUNC(x_unc_2bs, meas, bvals, grad_dirs));
    
    AIC_con_2zs = 2*N + K*log((1/K)*TwoZeppelinStickSSD(x_con_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
    AIC_unc_2zs = 2*N + K*log((1/K)*TwoZeppelinStickSSDUNC(x_unc_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
    
    if(bb==5)
        AIC_con_2zst = -1e4;
        AIC_unc_2zst = -1e4;
    else
        AIC_con_2zst = 2*N + K*log((1/K)*TwoZeppelinStickTortuositySSD(x_con_2zst, meas, bvals, grad_dirs, lambda1));
        AIC_unc_2zst = 2*N + K*log((1/K)*TwoZeppelinStickTortuositySSDUNC(x_unc_2zst, meas, bvals, grad_dirs, lambda1));
    end

    AIC_2bs = min(AIC_con_2bs, AIC_unc_2bs)
    AIC_2zs = min(AIC_con_2zs, AIC_unc_2zs)
    AIC_2zst = min(AIC_con_2zst, AIC_unc_2zst)
    
    BIC_con_2bs = N*log(K) + K*log((1/K)*TwoBallStickSSD(x_con_2bs, meas, bvals, grad_dirs));
    BIC_unc_2bs = N*log(K) + K*log((1/K)*TwoBallStickSSDUNC(x_unc_2bs, meas, bvals, grad_dirs));
    
    BIC_con_2zs = N*log(K) + K*log((1/K)*TwoZeppelinStickSSD(x_con_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
    BIC_unc_2zs = N*log(K) + K*log((1/K)*TwoZeppelinStickSSDUNC(x_unc_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
    
    if(bb==5)
        BIC_con_2zst = -1e4;
        BIC_unc_2zst = -1e4;
    else
        BIC_con_2zst = N*log(K) + K*log((1/K)*TwoZeppelinStickTortuositySSD(x_con_2zst, meas, bvals, grad_dirs, lambda1));
        BIC_unc_2zst = N*log(K) + K*log((1/K)*TwoZeppelinStickTortuositySSDUNC(x_unc_2zst, meas, bvals, grad_dirs, lambda1));
    end
    
    BIC_2bs = min(BIC_con_2bs, BIC_unc_2bs)
    BIC_2zs = min(BIC_con_2zs, BIC_unc_2zs)
    BIC_2zst = min(BIC_con_2zst, BIC_unc_2zst)


    AIC_bs_res(bb) = AIC_bs;
    AIC_zs_res(bb) = AIC_zs;
    AIC_zst_res(bb) = AIC_zst;
    BIC_bs_res(bb) = BIC_bs;
    BIC_zs_res(bb) = BIC_zs;
    BIC_zst_res(bb) = BIC_zst;
    AIC_2bs_res(bb) = AIC_2bs;
    AIC_2zs_res(bb) = AIC_2zs;
    AIC_2zst_res(bb) = AIC_2zst;
    BIC_2bs_res(bb) = BIC_2bs;
    BIC_2zs_res(bb) = BIC_2zs;
    BIC_2zst_res(bb) = BIC_2zst;

end
toc;

meas = D(:, 1); %Not to forget to switch back to first voxel

%% 

figure();
subplot(121)
plot(linspace(1, 6, 6), AIC_bs_res);
hold on;
plot(linspace(1, 6, 6), AIC_zs_res);
hold on;
plot(linspace(1, 6, 6), AIC_zst_res);
hold on;
plot(linspace(1, 6, 6), AIC_2bs_res);
hold on;
plot(linspace(1, 6, 6), AIC_2zs_res);
hold on;
plot(linspace(1, 6, 6), AIC_2zst_res);
hold off;
legend('Ball-and-stick', 'Zeppelin-and-stick', 'Zeppelin-and-stick with tortuosity', 'Ball-and-2-sticks', 'Zeppelin-and-2-sticks', 'Zeppelin-and-2-sticks with tortuosity');
xlabel('Voxel number');
ylabel('AIC value');
subplot(122)
plot(linspace(1, 6, 6), BIC_bs_res);
hold on;
plot(linspace(1, 6, 6), BIC_zs_res);
hold on;
plot(linspace(1, 6, 6), BIC_zst_res);
hold on;
plot(linspace(1, 6, 6), BIC_2bs_res);
hold on;
plot(linspace(1, 6, 6), BIC_2zs_res);
hold on;
plot(linspace(1, 6, 6), BIC_2zst_res);
hold off;
legend('Ball-and-stick', 'Zeppelin-and-stick', 'Zeppelin-and-stick with tortuosity', 'Ball-and-2-sticks', 'Zeppelin-and-2-sticks', 'Zeppelin-and-2-sticks with tortuosity');
xlabel('Voxel number');
ylabel('BIC value');
sgtitle('AIC and BIC values for different models and voxels');


%% Two balls and stick model

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoBallStickSSDOptim4(startx, meas, bvals, grad_dirs, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_2bs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_2bs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_2bs(1);
diff = x_con_2bs(2);
f1 = x_con_2bs(3);
theta1 = x_con_2bs(4);
phi1 = x_con_2bs(5);
f2 = x_con_2bs(6);
theta2 = x_con_2bs(7);
phi2 = x_con_2bs(8);
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoBallStickSSDOptim3(startx, meas, bvals, grad_dirs, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_2bs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_2bs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_2bs(1)^2;
diff = x_unc_2bs(2)^2;
f1 = exp(-(x_unc_2bs(3)^2));
theta1 = (exp(-(x_unc_2bs(4)^2)))*pi;
phi1 = (exp(-(x_unc_2bs(5)^2)))*2*pi;
f2 = exp(-(x_unc_2bs(6)^2));
theta2 = (exp(-(x_unc_2bs(7)^2)))*pi;
phi2 = (exp(-(x_unc_2bs(8)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
str2 = 'x1='+string(x_unc_2bs(1))+'; x2='+string(x_unc_2bs(2))+'; x3='+string(x_unc_2bs(3))+'; x4='+string(x_unc_2bs(4))+'; x5='+string(x_unc_2bs(5))+'; x6='+string(x_unc_2bs(6))+'; x7='+string(x_unc_2bs(7))+'; x8='+string(x_unc_2bs(8));
xlabel({str1, str2});

N = 8; %Not 5 anymore!
K = length(bvals);

AIC_con_2bs = 2*N + K*log((1/K)*TwoBallStickSSD(x_con_2bs, meas, bvals, grad_dirs));
AIC_unc_2bs = 2*N + K*log((1/K)*TwoBallStickSSDUNC(x_unc_2bs, meas, bvals, grad_dirs));
AIC_2bs = min(AIC_con_2bs, AIC_unc_2bs);
BIC_con_2bs = N*log(K) + K*log((1/K)*TwoBallStickSSD(x_con_2bs, meas, bvals, grad_dirs));
BIC_unc_2bs = N*log(K) + K*log((1/K)*TwoBallStickSSDUNC(x_unc_2bs, meas, bvals, grad_dirs));
BIC_2bs = min(BIC_unc_2bs, BIC_con_2bs);

% Apres faut recuperer tous les parametres de toutes les

%% Two sticks and zeppelin model

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoZeppelinStickSSDOptim4(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_2zs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_2zs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_2zs(1);
diff = x_con_2zs(2);
f1 = x_con_2zs(3);
theta1 = x_con_2zs(4);
phi1 = x_con_2zs(5);
f2 = x_con_2zs(6);
theta2 = x_con_2zs(7);
phi2 = x_con_2zs(8);
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoZeppelinStickSSDOptim3(startx, meas, bvals, grad_dirs, lambda1, lambda2, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_2zs = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_2zs = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_2zs(1)^2;
diff = x_unc_2zs(2)^2;
f1 = exp(-(x_unc_2zs(3)^2));
theta1 = (exp(-(x_unc_2zs(4)^2)))*pi;
phi1 = (exp(-(x_unc_2zs(5)^2)))*2*pi;
f2 = exp(-(x_unc_2zs(6)^2));
theta2 = (exp(-(x_unc_2zs(7)^2)))*pi;
phi2 = (exp(-(x_unc_2zs(8)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
str2 = 'x1='+string(x_unc_2zs(1))+'; x2='+string(x_unc_2zs(2))+'; x3='+string(x_unc_2zs(3))+'; x4='+string(x_unc_2zs(4))+'; x5='+string(x_unc_2zs(5))+'; x6='+string(x_unc_2zs(6))+'; x7='+string(x_unc_2zs(7))+'; x8='+string(x_unc_2zs(8));
xlabel({str1, str2});

N = 8; %Not 5 anymore!
K = length(bvals);

AIC_con_2zs = 2*N + K*log((1/K)*TwoZeppelinStickSSD(x_con_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
AIC_unc_2zs = 2*N + K*log((1/K)*TwoZeppelinStickSSDUNC(x_unc_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
AIC_2zs = min(AIC_con_2zs, AIC_unc_2zs);
BIC_con_2zs = N*log(K) + K*log((1/K)*TwoZeppelinStickSSD(x_con_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
BIC_unc_2zs = N*log(K) + K*log((1/K)*TwoZeppelinStickSSDUNC(x_unc_2zs, meas, bvals, grad_dirs, lambda1, lambda2));
BIC_2zs = min(BIC_unc_2zs, BIC_con_2zs);


%% Two sticks and zeppelin model WITH TORTUOSITY

% Parameters 
nb_of_retries = 5;
nb_of_steps = 23;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
startx_0 = [S0_lin d_lin_b 4.5e-01 1.0 1.0 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoZeppelinStickTortuositySSDOptim4(startx, meas, bvals, grad_dirs, lambda1, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_con_2zst = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_con_2zst = x;
            best_resnorm = RESNORM;
        end
    end
end

% Recovering the parameters from the output of fminunc
S0 = x_con_2zst(1);
diff = x_con_2zst(2);
f1 = x_con_2zst(3);
theta1 = x_con_2zst(4);
phi1 = x_con_2zst(5);
f2 = x_con_2zst(6);
theta2 = x_con_2zst(7);
phi2 = x_con_2zst(8);
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));

figure();
subplot(221);
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fmincon');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(222);
plot(meas, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (using fmincon)','SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
xlabel(str1);


res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-8,'TolFun',1e-8);

%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(4.5e-01)) 1.0 1.0 sqrt(-log(4.5e-01)) 1.0 1.0];


for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(0, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
        [x,RESNORM,~,~]=TwoZeppelinStickTortuositySSDOptim3(startx, meas, bvals, grad_dirs, lambda1, h);
        res_plot(ll, kk) = RESNORM;
        if(ll==1 && kk==1)
            x_unc_2zst = x;
            best_resnorm = RESNORM;
        end
        if(RESNORM<best_resnorm)
            x_unc_2zst = x;
            best_resnorm = RESNORM;
        end
    end
end

% Extract the parameters
S0 = x_unc_2zst(1)^2;
diff = x_unc_2zst(2)^2;
f1 = exp(-(x_unc_2zst(3)^2));
theta1 = (exp(-(x_unc_2zst(4)^2)))*pi;
phi1 = (exp(-(x_unc_2zst(5)^2)))*2*pi;
f2 = exp(-(x_unc_2zst(6)^2));
theta2 = (exp(-(x_unc_2zst(7)^2)))*pi;
phi2 = (exp(-(x_unc_2zst(8)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
S_est = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));

subplot(223);
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points using fminunc');
xlabel('Number of iterations');
ylabel('RESNORM');
subplot(224);
plot(meas, 'b+');
hold on;
plot(S, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (with fminunc)', 'SSD = '+string(best_resnorm)});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f1='+string(f1)+'; theta1='+string(theta1)+'; phi1='+string(phi1)+'; f2='+string(f2)+'; theta2='+string(theta2)+'; phi2='+string(phi2);
str2 = 'x1='+string(x_unc_2zst(1))+'; x2='+string(x_unc_2zst(2))+'; x3='+string(x_unc_2zst(3))+'; x4='+string(x_unc_2zst(4))+'; x5='+string(x_unc_2zst(5))+'; x6='+string(x_unc_2zst(6))+'; x7='+string(x_unc_2zst(7))+'; x8='+string(x_unc_2zst(8));
xlabel({str1, str2});

N = 8; %Not 5 anymore!
K = length(bvals);

AIC_con_2zst = 2*N + K*log((1/K)*TwoZeppelinStickTortuositySSD(x_con_2zst, meas, bvals, grad_dirs, lambda1));
AIC_unc_2zst = 2*N + K*log((1/K)*TwoZeppelinStickTortuositySSDUNC(x_unc_2zst, meas, bvals, grad_dirs, lambda1));
AIC_2zst = min(AIC_con_2zst, AIC_unc_2zst);
BIC_con_2zst = N*log(K) + K*log((1/K)*TwoZeppelinStickTortuositySSD(x_con_2zst, meas, bvals, grad_dirs, lambda1));
BIC_unc_2zst = N*log(K) + K*log((1/K)*TwoZeppelinStickTortuositySSDUNC(x_unc_2zst, meas, bvals, grad_dirs, lambda1));
BIC_2zst = min(BIC_unc_2zst, BIC_con_2zst);



%% Appendix - useful functions

% Function computing the mean diffusivity map
function md_value = md(D)
    [~, lambdas_d] = eig(D);
    md_value = (1/3) * (lambdas_d(1, 1) + lambdas_d(2, 2) + lambdas_d(3, 3));
end

% Function computing the FA map (gray-scaled, and directionnaly encoded
% with colours)
function [fa_value, fa_colour_value] = fa(D)
    [V, lambdas_d] = eig(D);
    fa_colour_value = abs(V(:, 1));
    lambdas = [lambdas_d(1, 1), lambdas_d(2, 2), lambdas_d(3, 3)];
    lambdas_mean = mean(lambdas);
    fa_map = sum((lambdas-lambdas_mean).^2) / sum(lambdas.^2);
    fa_value = sqrt( (3/2) * fa_map );
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim2(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim3(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim4(x, Avox, bvals, qhat, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickSSDOptim3(x, Avox, bvals, qhat, lambda1, lambda2, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@ZeppelinStickSSD, x, h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickSSDOptim4(x, Avox, bvals, qhat, lambda1, lambda2, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@ZeppelinStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickTortuositySSDOptim3(x, Avox, bvals, qhat, lambda1, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@ZeppelinStickSSD, x, h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickTortuositySSDOptim4(x, Avox, bvals, qhat, lambda1, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@ZeppelinStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoBallStickSSDOptim3(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoBallStickSSDOptim4(x, Avox, bvals, qhat, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f1 = x(3);
        theta1 = x(4);
        phi1 = x(5);
        f2 = x(6);
        theta2 = x(7);
        phi2 = x(8);
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoZeppelinStickSSDOptim3(x, Avox, bvals, qhat, lambda1, lambda2, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoZeppelinStickSSDOptim4(x, Avox, bvals, qhat, lambda1, lambda2, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f1 = x(3);
        theta1 = x(4);
        phi1 = x(5);
        f2 = x(6);
        theta2 = x(7);
        phi2 = x(8);
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoZeppelinStickTortuositySSDOptim3(x, Avox, bvals, qhat, lambda1, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoZeppelinStickTortuositySSDOptim4(x, Avox, bvals, qhat, lambda1, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f1 = x(3);
        theta1 = x(4);
        phi1 = x(5);
        f2 = x(6);
        theta2 = x(7);
        phi2 = x(8);
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function sumRes = BallStickSSD(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickSSD(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickTortuositySSD(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoBallStickSSD(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickSSD(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickTortuositySSD(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = BallStickSSDUNC(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickSSDUNC(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickTortuositySSDUNC(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoBallStickSSDUNC(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickSSDUNC(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickTortuositySSDUNC(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end
