clear;
close all;

%% Importing the data

load('data_p1/data.mat');
qhat = load('data_p1/bvecs');
bvals = 1000 * sum(qhat.*qhat);
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

%% 1.1.1 - Showing a sample from the data

% Array dimensions: dwis(image_volume; i; j; slice)

% Middle slice of the 1st image volume, which has b=0
figure();
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);
% Middle slice of the 2nd image volume, which has b=1000
figure();
imshow(flipud(squeeze(dwis(2,:,:,72))'), []);

%% 1.1.1 - Computing the D values for a particular voxel

% Let's only concentrate on one voxel at first
Avox = dwis(:, 92, 65, 72); % Voxel of interest
logAvox = log(Avox); % Applying the log
logAvox(isnan(logAvox)) = 0; % Processing
logAvox(logAvox<0) = 0;

% Let's construct the design matrix
G = zeros(108, 7); % We have 108 data points (rows) and 7 unknowns (columns)
G(:, 1) = ones(108, 1);
G(:, 2) = (-bvals .* (qhat(1, :).^2)).';
G(:, 3) = (-2*bvals .* (qhat(1, :).*qhat(2, :))).';
G(:, 4) = (-2*bvals .* (qhat(1, :).*qhat(3, :))).';
G(:, 5) = (-bvals .* (qhat(2, :).^2)).';
G(:, 6) = (-2*bvals .* (qhat(2, :).*qhat(3, :))).';
G(:, 7) = (-bvals .* (qhat(3, :).^2)).';

% Computing the x results vector
x_res = pinv(G) * logAvox;

% Recovering the elements of S(0,0) and D from x_res
logA0 = x_res(1);
D_res = [[x_res(2), x_res(3), x_res(4)], [x_res(3), x_res(5), x_res(6)], [x_res(4), x_res(6), x_res(7)]];

%% 1.1.1 - Computing the MD/FA maps for whole slices

% Voxel map of interest
vox72 = dwis(:, :, :, 72);
size_vox72 = size(vox72);
x_i = size_vox72(2);
y_j = size_vox72(3);

% Initializing our results maps
md72 = zeros(x_i, y_j);
fa72 = zeros(x_i, y_j);
rgb72 = zeros(x_i, y_j, 3);

for ii=1:x_i
    for jj=1:y_j

        voxOfInterest = vox72(:, ii, jj);
        %{
        logVox = log(voxOfInterest); % Log-map and processing
        logVox(isnan(logVox)) = 0; 
        logVox(logVox<0) = 0;
        %}
        
        if(min(voxOfInterest)>0)
            logVox = log(voxOfInterest);
            voxRes = pinv(G) * logVox; % Applying our model
            logVox0 = voxRes(1);
            voxD = [[voxRes(2) voxRes(3) voxRes(4)]; [voxRes(3) voxRes(5) voxRes(6)]; [voxRes(4) voxRes(6) voxRes(7)]];
            
            ouaisa = md(voxD);
            [ouaisb, ouaisc] = fa(voxD);
            md72(ii, jj) = (1/3)*(voxD(1,1)+voxD(2,2)+voxD(3,3));
            fa72(ii, jj) = ouaisb;
            rgb72(ii, jj, :) = ouaisc;
        end
        
    end
end

%% 1.1.1 - Visualizing the maps

figure();
subplot(131);
imshow(flipud((md72/max(md72(:))).'));
title('Mean diffusivity')
subplot(132);
imshow(flipud(fa72.'));
title('Fractional anisotropy')
subplot(133);
imshow(flipud(pagetranspose(rgb72)));
title('Fractional anisotropy (directionally encoded colour map)')



%% 1.1.2 - Testing the suggested routine

% Define a starting point for the non-linear fit
%startx = [3.5e+00 3e-03 2.5e-01 0 0];
startx = [3300 1.0e-03 4.5e-01 1.0 1.0];

% Define various options for the non-linear fitting algorithm.
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-10,'TolFun',1e-10);
% Now run the fitting
%[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=fminunc('BallStickSSDOptim', startx, Avox, bvals, qhat, h);
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=BallStickSSDOptim(startx, Avox, bvals, qhat, h);

% Recovering the parameters from the output of fminunc
S0 = parameter_hat(1);
diff = parameter_hat(2);
f = parameter_hat(3);
theta = parameter_hat(4);
phi = parameter_hat(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
% Estimated model
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
plot(Avox, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Unconstrained model','SSD = '+string(RESNORM/(1e6))+' e6'});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
xlabel(str1);


%% 1.1.3 - Adapting the implementation [without phi/theta constraints]
% We have modified BallStickSSDOptim into BallStickSSDOptim2 to take into account the restrictions on the parameters 

% Define a starting point for the non-linear fit
% USING THE INVERSE TRANSFORM: we are optimizing for x(1), x(2) etc... yes
% But if we want eventually to start from the same startx here, given that
% we use x(1)^2, x(2)^2.. we have to apply the transform BUT ONLY for startx
%startx = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];

% Define various options for the non-linear fitting algorithm.
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-10,'TolFun',1e-10);
% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=BallStickSSDOptim2(startx, Avox, bvals, qhat, h);

% Recovering the parameters from the output of fminunc
S0 = parameter_hat(1)^2;
diff = parameter_hat(2)^2;
f = exp(-(parameter_hat(3)^2));
theta = parameter_hat(4);
phi = parameter_hat(5);
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
% Estimated modeld
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
plot(Avox, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (but not phi/theta)','SSD = '+string(RESNORM/(1e6))+' e6'});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
str2 = 'x1='+string(parameter_hat(1))+'; x2='+string(parameter_hat(2))+'; x3='+string(parameter_hat(3))+'; x4='+string(parameter_hat(4))+'; x5='+string(parameter_hat(5));
xlabel({str1, str2});


%% 1.1.3 - Adapting the implementation [with phi/theta constraints]
% We have modified BallStickSSDOptim into BallStickSSDOptim3 to take into account the restrictions on the parameters 

% Define a starting point for the non-linear fit
%startx = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];

% Define various options for the non-linear fitting algorithm.
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-10,'TolFun',1e-10);
% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT]=BallStickSSDOptim3(startx, Avox, bvals, qhat, h);

% Recovering the parameters from the output of fminunc
S0 = parameter_hat(1)^2;
diff = parameter_hat(2)^2;
f = exp(-(parameter_hat(3)^2));
%theta = (exp(-(parameter_hat(4)^2))-0.5)*(pi/2);
%phi = (exp(-(parameter_hat(5)^2))-0.5)*pi;
theta = (exp(-(parameter_hat(4)^2)))*pi;
phi = (exp(-(parameter_hat(5)^2)))*2*pi;
% Synthesize the signals according to the model
fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
% Estimated model
S_est = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));

figure();
plot(Avox, 'b+');
hold on;
plot(S_est, 'rx');
hold off;
legend('Data', 'Estimated model');
title({'Constrained model (including phi/theta)', 'SSD = '+string(RESNORM/(1e6))+' e6'});
str1 = 'S0='+string(S0)+'; d='+string(diff)+'; f='+string(f)+'; theta='+string(theta)+'; phi='+string(phi);
str2 = 'x1='+string(parameter_hat(1))+'; x2='+string(parameter_hat(2))+'; x3='+string(parameter_hat(3))+'; x4='+string(parameter_hat(4))+'; x5='+string(parameter_hat(5));
xlabel({str1, str2});

%% 1.1.4 - Trying to get an intuition of the mean/variance of the data

% Let's try getting an intuition of the range of the parameters

% We have to use the only parameters we know
%startx = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-10,'TolFun',1e-10);

nb_of_samples_to_test = 40;
parameters_range = zeros(5, nb_of_samples_to_test);

for kk=1:nb_of_samples_to_test
    ATestVox = dwis(:, randi(145), randi(174), randi(145));
    [parameter_hat,~,~,~]=BallStickSSDOptim3(startx, ATestVox, bvals, qhat, h);
    for ii=1:5
        parameters_range(ii, kk) = parameter_hat(ii);
    end
end

for ii=1:5
    figure();
    plot(parameters_range(ii, :));
    switch ii
        case 1
            title('x(1)');
        case 2
            title('x(2)');
        case 3
            title('x(3)');
        case 4
            title('x(4)');
        case 5
            title('x(5)');
    end
end

%% 1.1.4 - Results of the previous cell

%{

x(1) varies between 0 and 100 with not too much intermediate values.
Let's take mu=20, var=1

x(2) mostly varies between 13 and 0
Let's take mu=3, var=1.5

x(3) mostly varies between 100 and 20 too
Let's take mu=40, var=1.5

x(4) and x(5), let's take mu=0, var=1

%}

%% 1.1.4 - Applying the noise and seeking the global minimum

% Parameters 
nb_of_retries = 5;
nb_of_steps = 15;
res_plot = zeros(nb_of_retries, nb_of_steps);
%h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6, 'Display', 'Iter');
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-7,'TolFun',1e-7);

%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(15, 1) normrnd(3.5, 1) normrnd(1, 1) normrnd(15, 1) normrnd(0, 1)];
        [~,RESNORM,~,~]=BallStickSSDOptim2(startx, Avox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

figure();
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points');
xlabel('Number of iterations');
ylabel('RESNORM');


%% 1.1.4 - Trying for different voxels

%{
DO WE HAVE TO WATCH OUT FOR NON-BRAIN VOXELS?
%}

nb_of_different_voxels = 10;
nb_of_steps = 10;
res_plot = zeros(nb_of_different_voxels, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-7,'TolFun',1e-7);

%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];

for ll=1:nb_of_different_voxels

    ATestVox = dwis(:, randi(145), randi(174), randi(145));
    while(not(min(ATestVox)>0)) % We check if this is a background 
        ATestVox = dwis(:, randi(145), randi(174), randi(145));
    end

    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(20, 1) normrnd(3.5, 1.5) normrnd(1, 1.5) normrnd(0, 1) normrnd(0, 1)];
        [~,RESNORM,~,~]=BallStickSSDOptim3(startx, ATestVox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

figure();
for ll=1:nb_of_different_voxels
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :))
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points and different voxels');
xlabel('Number of iterations');
ylabel('RESNORM');


%% 1.1.4

% To get an idea of the variations of each parameter, we will start by
% estimating the mean and variance for random samples of voxels
% We have 145*174*145 = 3 658 350 voxels

% Let's start by sampling random voxels, put them in an array, and then fit
% a normal distribution to this data
% Let's repeat this for some steps and average our results

% But to optimize, we start from the very same startx, pb?

% Apparently only intuition

%{
% Parameters of the sampling procedure
size_of_sampling = 500;
nb_of_steps_sampling = 10;

% Parameters for running the optimization
startx = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
h=optimset('MaxFunEvals',10000,'Algorithm','quasi-newton','TolX',1e-10,'TolFun',1e-10);

% 5 parameters to estimate
norm_estimate = zeros(5, 2, nb_of_steps_sampling);

% Iterating
for kk=1:nb_of_steps_sampling
    sample_sample = zeros(1, 500);
    for ii=1:500
        voxSample = dwis(:, randi(145), randi(174), randi(145));

    end
end
%}


%% 1.1.5

S072 = zeros(x_i, y_j);
d72 = zeros(x_i, y_j);
f72 = zeros(x_i, y_j);
resnorm72 = zeros(x_i, y_j);
phi72 = zeros(x_i, y_j);
theta72 = zeros(x_i, y_j);
nb_of_tries = 4; % Estimated thanks to the above parts

h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 0.5 0.8];
%startx = startx_0 + [normrnd(10, 1) normrnd(3.5, 1.5) normrnd(1, 1.5) normrnd(0, 1) normrnd(0, 1)];

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            % We start by computing a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx_0, voxOfInterest, bvals, qhat, h);
                   ouaisok = false;
                catch err
                   disp('Erreur normalement');
                end
            end
            best_params = parameter_hat;
            best_resnorm = RESNORM;
            for kk=2:nb_of_tries % Recomputing an estimate
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx_0 + [normrnd(20, 1) normrnd(3.5, 1.2) normrnd(1, 1.2) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx, voxOfInterest, bvals, qhat, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur normalement');
                    end
                end
                % If we achieve a better score, we store the new estimate, ow we discard it
                if(RESNORM<best_resnorm)
                    best_params = parameter_hat;
                    best_resnorm = RESNORM;
                end
            end
            S072(ii, jj) = best_params(1)^2;
            d72(ii, jj) = best_params(2)^2;
            f72(ii, jj) = exp(-(best_params(3)^2));
            theta72(ii, jj) = (exp(-(best_params(4)^2)))*pi;
            phi72(ii, jj) = (exp(-(best_params(5)^2)))*2*pi;
            resnorm72(ii, jj) = best_resnorm;
        end
    end
end
toc


%% 1.1.4 resnorm72 log

resnorm72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = resnorm72(ii, jj);
        if(ouais>0)
            resnorm72augm(ii, jj) = log(ouais);
        end
    end
end


%% 1.1.4 f72 discarded high values

d72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = d72(ii, jj);
        if(ouais<0.0105)
            d72augm(ii, jj) = ouais;
        end
    end
end


%% 1.1.4 Visualization of parameters (without phi and theta)

figure();
subplot(1, 4, 1);
imshow(flipud((S072/max(S072(:)))'), []);
title("S072");
subplot(1, 4, 2);
imshow(flipud((d72augm/max(d72augm(:)))'), []);
title("d72 (log-scaled)")
subplot(1, 4, 3);
imshow(flipud((f72/max(f72(:)))'), []);
title("f72")
subplot(1, 4, 4);
imshow(flipud((resnorm72augm/max(resnorm72augm(:)))'), []);
title("resnorm72 (log-scaled)")

% Quiver should not be pointing in that direction


%% 1.1.4 quiver map

%X = [1:x_i];
%Y = [1:y_j];
%U = cos(phi72)*sin(theta)

phi72tweak = phi72;
theta72tweak = theta72;
phi72tweak(phi72tweak>0.02) = phi72tweak(phi72tweak>0.02) - pi;
theta72tweak(theta72tweak>0.02) = theta72tweak(theta72tweak>0.02) - pi/2; 


figure();
subplot(121)
%flipud(quiver(cos(phi72).*sin(theta72), sin(phi72).*sin(theta72)));
%flipud(quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000));
quiver(cos(phi72tweak).*sin(theta72tweak).*1000, sin(phi72tweak).*sin(theta72tweak).*1000);
title('Fibre direction n')
subplot(122)
quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000);
title('Fibre direction n (scaled by f)')
set(gcf,'units','points','position',[10,10,1000,400])
%figure();
%flipud(quiver(cos(phi72).*sin(theta72).*f72, sin(phi72).*sin(theta72).*f72));

figure();
subplot(121)
%flipud(quiver(cos(phi72).*sin(theta72), sin(phi72).*sin(theta72)));
%flipud(quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000));
quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000);
title('Fibre direction n (scaled by f)')
subplot(122)
quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000);
title('Fibre direction n (scaled by f), zoomed in')
set(gcf,'units','points','position',[10,10,1000,400])


%cos(phi)*sin(theta) sin(phi)*sin(theta)


%% 1.1.5 Optimisation with fmincon - error plots

% Parameters 
nb_of_retries = 3;
nb_of_steps = 20;
res_plot = zeros(nb_of_retries, nb_of_steps);
h=optimset('MaxFunEvals',20000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8);
%h=optimset('MaxFunEvals',20000,'Algorithm','trust-region-reflective','GradObj','on','TolX',1e-10,'TolFun',1e-10);

%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(15, 1) normrnd(3.5, 1) normrnd(1, 1) normrnd(15, 1) normrnd(0, 1)];
        [~,RESNORM,~,~]=BallStickSSDOptim4(startx, Avox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

figure();
subplot(121)
for ll=1:nb_of_retries
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points');
xlabel('Number of iterations');
ylabel('RESNORM');

nb_of_different_voxels = 10;
nb_of_steps = 10;
res_plot = zeros(nb_of_different_voxels, nb_of_steps);

startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];

for ll=1:nb_of_different_voxels

    ATestVox = dwis(:, randi(145), randi(174), randi(145));
    while(not(min(ATestVox)>0)) % We check if this is a background 
        ATestVox = dwis(:, randi(145), randi(174), randi(145));
    end

    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(20, 1) normrnd(3.5, 1.5) normrnd(1, 1.5) normrnd(0, 1) normrnd(0, 1)];
        [~,RESNORM,~,~]=BallStickSSDOptim4(startx, ATestVox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

subplot(122)
for ll=1:nb_of_different_voxels
    %plot(res_plot(ll, :));
    semilogy(res_plot(ll, :))
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points and different voxels');
xlabel('Number of iterations');
ylabel('RESNORM');

% La faire un subplot avec le truc au-dessus et à côté le truc qui fait
% pour différents voxels


%% 1.1.5 Optimization with fmincon - whole slice

S072 = zeros(x_i, y_j);
d72 = zeros(x_i, y_j);
f72 = zeros(x_i, y_j);
resnorm72 = zeros(x_i, y_j);
phi72 = zeros(x_i, y_j);
theta72 = zeros(x_i, y_j);
nb_of_tries = 4; % Estimated thanks to the above parts

h=optimset('MaxFunEvals',9000,'Algorithm','interior-point','TolX',5e-6,'TolFun',5e-6);
startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
%startx = startx_0 + [normrnd(10, 1) normrnd(3.5, 1.5) normrnd(1, 1.5) normrnd(0, 1) normrnd(0, 1)];

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            % We start by computing a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=BallStickSSDOptim4(startx_0, voxOfInterest, bvals, qhat, h);
                   ouaisok = false;
                catch err
                   disp('Erreur normalement');
                end
            end
            best_params = parameter_hat;
            best_resnorm = RESNORM;
            for kk=2:nb_of_tries % Recomputing an estimate
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx_0 + [normrnd(20, 1) normrnd(3.5, 1.2) normrnd(1, 1.2) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=BallStickSSDOptim4(startx, voxOfInterest, bvals, qhat, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur normalement');
                    end
                end
                % If we achieve a better score, we store the new estimate, ow we discard it
                if(RESNORM<best_resnorm)
                    best_params = parameter_hat;
                    best_resnorm = RESNORM;
                end
            end
            S072(ii, jj) = best_params(1);
            d72(ii, jj) = best_params(2);
            f72(ii, jj) = best_params(3);
            theta72(ii, jj) = (exp(-(best_params(4)^2)))*pi;
            phi72(ii, jj) = (exp(-(best_params(5)^2)))*2*pi;
            resnorm72(ii, jj) = best_resnorm;
        end
    end
end
toc

%% 1.1.5 plotting

resnorm72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = resnorm72(ii, jj);
        if(ouais>0)
            resnorm72augm(ii, jj) = log(ouais);
        end
    end
end

d72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = d72(ii, jj);
        %if(ouais<0.0105)
        if(ouais<2.481610112057668e-05)
            d72augm(ii, jj) = ouais;
        end
    end
end

figure();
subplot(1, 4, 1);
imshow(flipud((S072/max(S072(:)))'), []);
title("S072");
subplot(1, 4, 2);
%imshow(flipud((d72augm/max(d72augm(:)))'), []);
imshow(flipud((d72/max(d72(:)))'), []);
title("d72 (log-scaled)")
subplot(1, 4, 3);
imshow(flipud((f72/max(f72(:)))'), []);
title("f72")
subplot(1, 4, 4);
imshow(flipud((resnorm72augm/max(resnorm72augm(:)))'), []);
title("resnorm72 (log-scaled)")

% Quiver should not be pointing in that direction

phi72tweak = phi72;
theta72tweak = theta72;
%phi72tweak(phi72tweak>0.02) = phi72tweak(phi72tweak>0.02) - pi;
%theta72tweak(theta72tweak>0.02) = theta72tweak(theta72tweak>0.02) - pi/2; 


figure();
subplot(121)
%flipud(quiver(cos(phi72).*sin(theta72), sin(phi72).*sin(theta72)));
%flipud(quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000));
quiver(cos(phi72tweak).*sin(theta72tweak).*1000, sin(phi72tweak).*sin(theta72tweak).*1000);
title('Fibre direction n')
subplot(122)
quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000);
title('Fibre direction n (scaled by f)')
set(gcf,'units','points','position',[10,10,1000,400])
%figure();
%flipud(quiver(cos(phi72).*sin(theta72).*f72, sin(phi72).*sin(theta72).*f72));


%% 1.1.6 fminunc with gradient

% Desgin matrix for linear estimation
G = [ones(1, length(bvals)); -bvals].';
% We start by estimating S0 and d
x = pinv(G)*log(Avox); 
S0_lin = max(2500, exp(x(1)));
d_lin = max(0, x(2));

% Parameters 
nb_of_retries = 5;
nb_of_steps = 15;
res_plot = zeros(nb_of_retries, nb_of_steps);
%h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6, 'Display', 'Iter');
%h=optimset('MaxFunEvals',20000,'Algorithm','trust-region','TolX',1e-9,'TolFun',1e-9,'GradObj', 'on','DerivativeCheck','on');
%h=optimset('MaxFunEvals',30000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'DerivativeCheck','on', 'GradObj', 'on');
h=optimset('MaxFunEvals',30000,'Algorithm','trust-region-reflective','TolX',1e-9,'TolFun',1e-9,'GradObj', 'on');
%h=optimset('MaxFunEvals',20000,'Algorithm','active-set','TolX',1e-8,'TolFun',1e-8,'DerivativeCheck','on', 'GradObj', 'on');

%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];
%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
        [ouaiscquoi,RESNORM,~,~]=BallStickSSDOptim4GRAD(startx, Avox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

figure();
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points');
xlabel('Number of iterations');
ylabel('RESNORM');


%% 1.1.6 - informed starting points

S072_i = zeros(x_i, y_j);
d72_i = zeros(x_i, y_j);
f72_i = zeros(x_i, y_j);
resnorm72_i = zeros(x_i, y_j);
phi72_i = zeros(x_i, y_j);
theta72_i = zeros(x_i, y_j);

% Desgin matrix for linear estimation
G = [ones(1, length(bvals)); -bvals].';

nb_of_tries = 10; % Estimated thanks to the above parts
h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(5e-01)) 1 1];

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            % We start by estimating S0 and d
            x = pinv(G)*log(voxOfInterest); 
            S0_lin = max(2500, exp(x(1)));
            d_lin = max(0, x(2));

            startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];

            % Then we can compute a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx_0, voxOfInterest, bvals, qhat, h);
                   ouaisok = false;
                catch err
                   disp('Erreur normalement');
                end
            end
            best_params = parameter_hat;
            best_resnorm = RESNORM;
            for kk=2:nb_of_tries % Recomputing an estimate
                if(best_resnorm<(6e6))
                    break;
                end
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx, voxOfInterest, bvals, qhat, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur normalement');
                    end
                end
                % If we achieve a better score, we store the new estimate, ow we discard it
                if(RESNORM<best_resnorm)
                    best_params = parameter_hat;
                    best_resnorm = RESNORM;
                end
            end
            S072_i(ii, jj) = best_params(1)^2;
            d72_i(ii, jj) = best_params(2)^2;
            f72_i(ii, jj) = exp(-(best_params(3)^2));
            theta72_i(ii, jj) = (exp(-(best_params(4)^2)))*pi;
            phi72_i(ii, jj) = (exp(-(best_params(5)^2)))*2*pi;
            resnorm72_i(ii, jj) = best_resnorm;
        end
    end
end
toc
disp('Version with informed starting points');


%% 1.1.6 - PLOTTING informed starting points

resnorm72augm_i = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = resnorm72_i(ii, jj);
        if(ouais>0)
            resnorm72augm_i(ii, jj) = log(ouais);
        end
    end
end

d72augm_i = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = d72_i(ii, jj);
        if(ouais<0.0105)
        %if(ouais<2.481610112057668e-05)
            d72augm_i(ii, jj) = ouais;
        end
    end
end

figure();
subplot(1, 4, 1);
imshow(flipud((S072_i/max(S072_i(:)))'), []);
title("S072");
subplot(1, 4, 2);
imshow(flipud((d72augm_i/max(d72augm_i(:)))'), []);
%imshow(flipud((d72_i/max(d72_i(:)))'), []);
title("d72 (log-scaled)")
subplot(1, 4, 3);
imshow(flipud((f72_i/max(f72_i(:)))'), []);
title("f72")
subplot(1, 4, 4);
imshow(flipud((resnorm72augm_i/max(resnorm72augm_i(:)))'), []);
title("resnorm72 (log-scaled)")

% Quiver should not be pointing in that direction

phi72tweak_i = phi72_i;
theta72tweak_i = theta72_i;
phi72tweak_i(phi72tweak_i>0.02) = phi72tweak_i(phi72tweak_i>0.02) - pi;
theta72tweak_i(theta72tweak_i>0.02) = theta72tweak_i(theta72tweak_i>0.02) - pi/2; 


figure();
subplot(121)
quiver(cos(phi72tweak_i).*sin(theta72tweak_i).*1000, sin(phi72tweak_i).*sin(theta72tweak_i).*1000);
title('Fibre direction n')
subplot(122)
quiver(cos(phi72tweak_i).*sin(theta72tweak_i).*f72_i.*1000, sin(phi72tweak_i).*sin(theta72tweak_i).*f72_i.*1000);
title('Fibre direction n (scaled by f)')
set(gcf,'units','points','position',[10,10,1000,400])


%% 1.1.6 not informed pts but still adapted with resnorm

S072 = zeros(x_i, y_j);
d72 = zeros(x_i, y_j);
f72 = zeros(x_i, y_j);
resnorm72 = zeros(x_i, y_j);
phi72 = zeros(x_i, y_j);
theta72 = zeros(x_i, y_j);

% Desgin matrix for linear estimation
G = [ones(1, length(bvals)); -bvals].';

nb_of_tries = 10; % Estimated thanks to the above parts
h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(5e-01)) 1 1];

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels

            % We start by computing a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx_0, voxOfInterest, bvals, qhat, h);
                   ouaisok = false;
                catch err
                   disp('Erreur normalement');
                end
            end
            best_params = parameter_hat;
            best_resnorm = RESNORM;
            for kk=2:nb_of_tries % Recomputing an estimate
                if(best_resnorm<(6e6))
                    break;
                end
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=BallStickSSDOptim3(startx, voxOfInterest, bvals, qhat, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur normalement');
                    end
                end
                % If we achieve a better score, we store the new estimate, ow we discard it
                if(RESNORM<best_resnorm)
                    best_params = parameter_hat;
                    best_resnorm = RESNORM;
                end
            end
            S072(ii, jj) = best_params(1)^2;
            d72(ii, jj) = best_params(2)^2;
            f72(ii, jj) = exp(-(best_params(3)^2));
            theta72(ii, jj) = (exp(-(best_params(4)^2)))*pi;
            phi72(ii, jj) = (exp(-(best_params(5)^2)))*2*pi;
            resnorm72(ii, jj) = best_resnorm;
        end
    end
end
toc
disp('Version with NON-informed starting points');


%% 1.1.6 - PLOTTING not informed pts but still adapted with resnorm

resnorm72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = resnorm72(ii, jj);
        if(ouais>0)
            resnorm72augm(ii, jj) = log(ouais);
        end
    end
end

d72augm = zeros(x_i, y_j);
for ii=1:x_i
    for jj=1:y_j
        ouais = d72(ii, jj);
        if(ouais<0.0105)
        %if(ouais<2.481610112057668e-05)
            d72augm(ii, jj) = ouais;
        end
    end
end

figure();
subplot(1, 4, 1);
imshow(flipud((S072/max(S072(:)))'), []);
title("S072");
subplot(1, 4, 2);
imshow(flipud((d72augm/max(d72augm(:)))'), []);
%imshow(flipud((d72/max(d72(:)))'), []);
title("d72 (log-scaled)")
subplot(1, 4, 3);
imshow(flipud((f72/max(f72(:)))'), []);
title("f72")
subplot(1, 4, 4);
imshow(flipud((resnorm72augm/max(resnorm72augm(:)))'), []);
title("resnorm72 (log-scaled)")

% Quiver should not be pointing in that direction

phi72tweak = phi72;
theta72tweak = theta72;
phi72tweak(phi72tweak>0.02) = phi72tweak(phi72tweak>0.02) - pi;
theta72tweak(theta72tweak>0.02) = theta72tweak(theta72tweak>0.02) - pi/2; 


figure();
subplot(121)
quiver(cos(phi72tweak).*sin(theta72tweak).*1000, sin(phi72tweak).*sin(theta72tweak).*1000);
title('Fibre direction n')
subplot(122)
quiver(cos(phi72tweak).*sin(theta72tweak).*f72.*1000, sin(phi72tweak).*sin(theta72tweak).*f72.*1000);
title('Fibre direction n (scaled by f)')
set(gcf,'units','points','position',[10,10,1000,400])


%% 1.1.6 - Error plots for voxels for informed points

% Parameters 
nb_of_retries = 5;
nb_of_steps = 15;
res_plot = zeros(nb_of_retries, nb_of_steps);
%h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6, 'Display', 'Iter');
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-7,'TolFun',1e-7);

%startx_0 = [sqrt(3.5e+00) sqrt(3e-03) sqrt(-log(2.5e-01)) 0 0];
%startx_0 = [sqrt(3300) sqrt(1.0e-03) sqrt(-log(4.5e-01)) 1.0 1.0];
x = pinv(G)*log(Avox); 
S0_lin = max(2500, exp(x(1)));
d_lin = max(0, x(2));
startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];

for ll=1:nb_of_retries
    for kk=1:nb_of_steps
        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
        [~,RESNORM,~,~]=BallStickSSDOptim2(startx, Avox, bvals, qhat, h);
        res_plot(ll, kk) = RESNORM;
    end
end

figure();
for ll=1:nb_of_retries
    plot(res_plot(ll, :));
    hold on;
end
hold off;
title('Evolution of the RESNORM for different starting points');
xlabel('Number of iterations');
ylabel('RESNORM');


%% 1.2.1 Bootstrap procedure

% Voxel used in part 1.1
Avox = dwis(:, 92, 65, 72); % Voxel of interest
logAvox = log(Avox); % Applying the log
logAvox(isnan(logAvox)) = 0; % Processing
logAvox(logAvox<0) = 0;

% What we will do, is create a new voxel based on Avox, we will sample
% randomly WITH replacement, and perform optimization on this to retrieve
% the parameters. We will then plot an histogram of the values we obtain
% (that should look like a Gaussian) to determine the 2-sigma range
% We will use the added noise routine to try to fall into the global
% minimum. We will also use a condition that makes the iteration stop once
% we fall < 6e6 for the RESNORM (bc we know that is the global minimum)

% Parameters 
bootstrap_steps = 500;
S0_boot = zeros(bootstrap_steps, 1);
d_boot = zeros(bootstrap_steps, 1);
f_boot = zeros(bootstrap_steps, 1);
theta_boot = zeros(bootstrap_steps, 1);
phi_boot = zeros(bootstrap_steps, 1);
nb_of_optim_steps = 5;
h=optimset('MaxFunEvals',15000,'Algorithm','interior-point','TolX',1e-9,'TolFun',1e-9, 'Display', 'off');

%startx_0 = [3300 1.0e-03 4.5e-01 1.0 1.0];
% Desgin matrix for linear estimation
G = [ones(1, length(bvals)); -bvals].';
x = pinv(G)*log(Avox); 
S0_dlin = exp(x(1));
d_dlin = x(2);
S0_lin = max(2500, S0_dlin);
d_lin = max(0, d_dlin);

startx_0 = [S0_lin d_lin 4.5e-01 1.0 1.0];

for kk=1:bootstrap_steps

    % First we build the bootstrap sample
    Avox_bootstrap = zeros(size(Avox, 1), 1);
    for boot_idx=1:size(Avox, 1)
        Avox_bootstrap(boot_idx, 1) = Avox(randi(size(Avox, 1)), 1);
    end
    
    % Then we run the optimization
    for retry_idx=1:nb_of_optim_steps
        %startx = startx_0 + [normrnd(15, 1) normrnd(3.5, 1) normrnd(0, 1) normrnd(0, 1) normrnd(0, 1)];
        startx = startx_0 + [normrnd(0, 10) normrnd(0, 0.5) normrnd(0.3, 0.25) normrnd(0, 1) normrnd(0, 1)];
        [parameter_hat,RESNORM,~,~]=BallStickSSDOptim4(startx, Avox_bootstrap, bvals, qhat, h);
        if(RESNORM<(6e6))
            break;
        end
    end

    % Appending the results to our arrays
    S0_boot(kk, 1) = parameter_hat(1);
    d_boot(kk, 1) = parameter_hat(2);
    f_boot(kk, 1) = parameter_hat(3);
    theta_boot(kk, 1) = parameter_hat(4);
    phi_boot(kk, 1) = parameter_hat(5);

end

figure();
subplot(2, 3, 1);
histogram(S0_boot);
title('S0');
subplot(2, 3, 2);
histogram(d_boot);
title('d');
subplot(2, 3, 3);
histogram(f_boot);
title('f');
subplot(2, 3, 4.5);
histogram(theta_boot);
title('theta');
subplot(2, 3, 5.5);
histogram(phi_boot);
title('phi');
sgtitle('Histograms of the bootstrapped parameter values');


%% 1.2.2 MCMC/Metropolis-Hastings

% Parameters 
mcmc_steps = 10000;
x_mcmc = zeros(mcmc_steps, 5);

% Desgin matrix for linear estimation
G = [ones(1, length(bvals)); -bvals].';
x = pinv(G)*log(Avox); 
S0_dlin = exp(x(1));
d_dlin = x(2);
S0_lin = max(2500, S0_dlin);
d_lin = max(0, d_dlin);


%h=optimset('MaxFunEvals',15000,'Algorithm','interior-point','TolX',1e-9,'TolFun',1e-9, 'Display', 'off');
x_1 = [4000 5.0e-02 5e-01 1.0 1.0];
%x_1 = [S0_lin, d_lin, 4.5e-01 1.0 1.0];
%x_1 = [2400 1 0.5 0 0];
like_x_1 = BallStickSSD(x_1, Avox, bvals, qhat);
x_mcmc(1, :) = x_1;


acceptance_rate = 0;
ratio_like_arr = zeros(mcmc_steps, 1);
like_x_1_arr = zeros(mcmc_steps, 1);
like_x_2_arr = zeros(mcmc_steps, 1);
log_alpha_arr = zeros(mcmc_steps, 1);
alpha_arr = zeros(mcmc_steps, 1);

mu_S0 = 0;
std_S0 = 25;
mu_d = 0;
std_d = 0.6;
mu_f = 0;
std_f = 0.02;
mu_theta = 0;
std_theta = 0.3;
mu_phi = 0;
std_phi = 0.3;

for kk=2:mcmc_steps
    
    %x_2 = x_1 + [normrnd(mu_S0, std_S0) normrnd(mu_d, std_d) normrnd(mu_f, std_f) normrnd(mu_theta, std_theta) normrnd(mu_phi, std_phi)]; % Our perturbation
    x_2 = x_1 + [normrnd(mu_S0, std_S0/10) normrnd(mu_d, std_d/10) normrnd(mu_f, std_f/10) normrnd(mu_theta, std_theta/10) normrnd(mu_phi, std_phi/10)]; 
    like_x_2 = BallStickSSD(x_2, Avox, bvals, qhat);
    while(like_x_2==Inf || isnan(like_x_2) || like_x_2>1e11)
        x_2 = x_1 + [normrnd(mu_S0, std_S0) normrnd(mu_d, std_d) normrnd(mu_f, std_f) normrnd(mu_theta, std_theta) normrnd(mu_phi, std_phi)]; % Our perturbation
        %x_2 = x_1 + [normrnd(mu_S0, std_S0/4) normrnd(mu_d, std_d/4) normrnd(mu_f, std_f/4) normrnd(mu_theta, std_theta/4) normrnd(mu_phi, std_phi/4)]; % Our perturbation
        like_x_2 = BallStickSSD(x_2, Avox, bvals, qhat);
    end

    format long g;
    factor1 = 1e20;
    factor2 = 1e6;
    sigma = 220;
    factor = 1/(1000*(sigma.^2));
    ratio_like = vpa(exp((like_x_2*factor)-(like_x_1*factor)));
    %ratio_like = (like_x_2-like_x_1)/factor;
    %ratio_like = like_x_1/like_x_2;
    like_x_1_arr(kk) = like_x_1;
    like_x_2_arr(kk) = like_x_2;
    ratio_like_arr(kk) = ratio_like;

    log_alpha = log(unifrnd(0, 1));
    log_alpha_arr(kk) = log_alpha;
    alpha = unifrnd(0, 1);
    alpha_arr(kk) = alpha;
    %if(ratio_like > log_alpha)
    if(min(1, ratio_like) > alpha)
        x_1 = x_2;
        like_x_1 = like_x_2;
        acceptance_rate = acceptance_rate + 1;
    end

    x_mcmc(kk, :) = x_1;

end

figure();
subplot(2, 5, 1);
histogram(x_mcmc(:, 1));
title('S0');
subplot(2, 5, 2);
histogram(x_mcmc(:, 2));
title('d');
subplot(2, 5, 3);
histogram(x_mcmc(:, 3));
title('f');
subplot(2, 5, 4);
histogram(x_mcmc(:, 4));
title('theta');
subplot(2, 5, 5);
histogram(x_mcmc(:, 5));
title('phi');
subplot(2, 5, 6);
plot(x_mcmc(:, 1));
subplot(2, 5, 7);
plot(x_mcmc(:, 2));
subplot(2, 5, 8);
plot(x_mcmc(:, 3));
subplot(2, 5, 9);
plot(x_mcmc(:, 4));
subplot(2, 5, 10);
plot(x_mcmc(:, 5));
str1 = 'Histograms of the MCMC parameter values (top), and chains (bottom)';
str2 = 'Acceptance rate: '+string(acceptance_rate/mcmc_steps);
sgtitle({str1, str2});

%figure();
%histogram(ratio_like_arr);



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

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
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
        %theta = (exp(-(x(4)^2))-0.5)*(pi/2);
        %phi = (exp(-(x(5)^2))-0.5)*pi;
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

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim4GRAD(x, Avox, bvals, qhat, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function [sumRes, sumResDeriv] = BallStickSSD(x)
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
        if nargout > 1 % gradient required
            sumResDerivS0 = f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff);
            sumResDerivdiff = S0*(-bvals.*(fibdotgrad.^2).*f.*exp(-bvals*diff.*(fibdotgrad.^2)) -bvals*(1-f).*exp(-bvals*diff));
            sumResDerivf = S0*(exp(-bvals*diff.*(fibdotgrad.^2)) - exp(-bvals*diff));

            %theta_inter = (cos(phi).^2)*2*sin(theta)*cos(theta).*(qhat(1, :).^2) + 2*(sin(phi).^2)*cos(theta)*sin(theta).*(qhat(2, :).^2) - 2*cos(theta)*sin(theta).*qhat(3, :) + 4*cos(phi)*sin(phi)*cos(theta)*sin(theta).*qhat(1, :).*qhat(2, :) + 2*cos(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(1, :).*qhat(3, :) + 2*sin(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(2, :).*qhat(3, :);
            %phi_inter = -2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(1, :).^2) + 2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(2, :).^2) + 2*((cos(phi).^2) - (sin(phi).^2))*(sin(theta).^2).*qhat(1, :).*qhat(2, :) -2*sin(phi)*sin(theta)*cos(theta).*qhat(1, :).*qhat(3, :) +2*cos(phi)*sin(theta)*cos(theta).*qhat(2, :).*qhat(3, :);
            %sumResDerivtheta = S0*(-f*theta_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));
            %sumResDerivphi = S0*(-f*phi_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));

            fibdir_theta = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
            fibdir_phi = [-sin(phi)*sin(theta) cos(phi)*sin(theta) 0.0];
            %fibdotgrad_theta = sum((qhat.^2).*repmat(fibdir_theta, [length(qhat) 1])');
            %fibdotgrad_phi = sum((qhat.^2).*repmat(fibdir_phi, [length(qhat) 1])');
            fibdotgrad_theta = sum((qhat).*repmat(fibdir_theta, [length(qhat) 1])');
            fibdotgrad_phi = sum((qhat).*repmat(fibdir_phi, [length(qhat) 1])');
            sumResDerivtheta = S0*(-2*bvals*diff.*fibdotgrad_theta.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            sumResDerivphi = S0*(-2*bvals*diff.*fibdotgrad_phi.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));

            sumResDeriv = zeros(5, 1);
            sumResDeriv(1) = sum(-2*sumResDerivS0*(Avox - S'));
            sumResDeriv(2) = sum(-2*sumResDerivdiff*(Avox - S'));
            sumResDeriv(3) = sum(-2*sumResDerivf*(Avox - S'));
            sumResDeriv(4) = sum(-2*sumResDerivtheta*(Avox - S'));
            sumResDeriv(5) = sum(-2*sumResDerivphi*(Avox - S'));        
            %{
            sumResDeriv(1) = -((Avox - S').')*sumResDerivS0 - (sumResDerivS0.')*(Avox - S');
            sumResDeriv(2) = -((Avox - S').')*sumResDerivdiff - (sumResDerivdiff.')*(Avox - S');
            sumResDeriv(3) = -((Avox - S').')*sumResDerivf - (sumResDerivf.')*(Avox - S');
            sumResDeriv(4) = -((Avox - S').')*sumResDerivtheta - (sumResDerivtheta.')*(Avox - S');
            sumResDeriv(5) = -((Avox - S').')*sumResDerivphi - (sumResDerivphi.')*(Avox - S');  
            %}

        end
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

%{

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim2GRAD(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function [sumRes, sumResDeriv] = BallStickSSD(x)
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
        if nargout > 1 % gradient required
            sumResDerivS0 = f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff);
            sumResDerivdiff = S0*(-bvals.*(fibdotgrad.^2).*f.*exp(-bvals*diff.*(fibdotgrad.^2)) -bvals*(1-f).*exp(-bvals*diff));
            sumResDerivf = S0*(exp(-bvals*diff.*(fibdotgrad.^2)) - exp(-bvals*diff));

            %theta_inter = (cos(phi).^2)*2*sin(theta)*cos(theta).*(qhat(1, :).^2) + 2*(sin(phi).^2)*cos(theta)*sin(theta).*(qhat(2, :).^2) - 2*cos(theta)*sin(theta).*qhat(3, :) + 4*cos(phi)*sin(phi)*cos(theta)*sin(theta).*qhat(1, :).*qhat(2, :) + 2*cos(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(1, :).*qhat(3, :) + 2*sin(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(2, :).*qhat(3, :);
            %phi_inter = -2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(1, :).^2) + 2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(2, :).^2) + 2*((cos(phi).^2) - (sin(phi).^2))*(sin(theta).^2).*qhat(1, :).*qhat(2, :) -2*sin(phi)*sin(theta)*cos(theta).*qhat(1, :).*qhat(3, :) +2*cos(phi)*sin(theta)*cos(theta).*qhat(2, :).*qhat(3, :);
            %sumResDerivtheta = S0*(-f*theta_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));
            %sumResDerivphi = S0*(-f*phi_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));

            fibdir_theta = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
            fibdir_phi = [-sin(phi)*sin(theta) cos(phi)*sin(theta) 0.0];
            %fibdotgrad_theta = sum((qhat.^2).*repmat(fibdir_theta, [length(qhat) 1])');
            %fibdotgrad_phi = sum((qhat.^2).*repmat(fibdir_phi, [length(qhat) 1])');
            fibdotgrad_theta = sum((qhat).*repmat(fibdir_theta, [length(qhat) 1])');
            fibdotgrad_phi = sum((qhat).*repmat(fibdir_phi, [length(qhat) 1])');
            sumResDerivtheta = S0*(-2*bvals*diff.*fibdotgrad_theta.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            sumResDerivphi = S0*(-2*bvals*diff.*fibdotgrad_phi.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));

            sumResDeriv = zeros(5, 1);
            sumResDeriv(1) = sum(-2*sumResDerivS0*(Avox - S'));
            sumResDeriv(2) = sum(-2*sumResDerivdiff*(Avox - S'));
            sumResDeriv(3) = sum(-2*sumResDerivf*(Avox - S'));
            sumResDeriv(4) = sum(-2*sumResDerivtheta*(Avox - S'));
            sumResDeriv(5) = sum(-2*sumResDerivphi*(Avox - S'));        
            %{
            sumResDeriv(1) = -((Avox - S').')*sumResDerivS0 - sumResDerivS0*(Avox - S');
            sumResDeriv(2) = -((Avox - S').')*sumResDerivdiff - sumResDerivdiff*(Avox - S');
            sumResDeriv(3) = -((Avox - S').')*sumResDerivf - sumResDerivf*(Avox - S');
            sumResDeriv(4) = -((Avox - S').')*sumResDerivtheta - sumResDerivtheta*(Avox - S');
            sumResDeriv(5) = -((Avox - S').')*sumResDerivphi - sumResDerivphi*(Avox - S');       
            %}
        end
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim3GRAD(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function [sumRes, sumResDeriv] = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        %theta = (exp(-(x(4)^2))-0.5)*(pi/2);
        %phi = (exp(-(x(5)^2))-0.5)*pi;
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
        if nargout > 1 % gradient required
            sumResDerivS0 = f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff);
            sumResDerivdiff = S0*(-bvals.*(fibdotgrad.^2).*f.*exp(-bvals*diff.*(fibdotgrad.^2)) -bvals*(1-f).*exp(-bvals*diff));
            sumResDerivf = S0*(exp(-bvals*diff.*(fibdotgrad.^2)) - exp(-bvals*diff));

            %theta_inter = (cos(phi).^2)*2*sin(theta)*cos(theta).*(qhat(1, :).^2) + 2*(sin(phi).^2)*cos(theta)*sin(theta).*(qhat(2, :).^2) - 2*cos(theta)*sin(theta).*qhat(3, :) + 4*cos(phi)*sin(phi)*cos(theta)*sin(theta).*qhat(1, :).*qhat(2, :) + 2*cos(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(1, :).*qhat(3, :) + 2*sin(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(2, :).*qhat(3, :);
            %phi_inter = -2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(1, :).^2) + 2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(2, :).^2) + 2*((cos(phi).^2) - (sin(phi).^2))*(sin(theta).^2).*qhat(1, :).*qhat(2, :) -2*sin(phi)*sin(theta)*cos(theta).*qhat(1, :).*qhat(3, :) +2*cos(phi)*sin(theta)*cos(theta).*qhat(2, :).*qhat(3, :);
            %sumResDerivtheta = S0*(-f*theta_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));
            %sumResDerivphi = S0*(-f*phi_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));

            fibdir_theta = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
            fibdir_phi = [-sin(phi)*sin(theta) cos(phi)*sin(theta) 0.0];
            %fibdotgrad_theta = sum((qhat.^2).*repmat(fibdir_theta, [length(qhat) 1])');
            %fibdotgrad_phi = sum((qhat.^2).*repmat(fibdir_phi, [length(qhat) 1])');
            fibdotgrad_theta = sum((qhat).*repmat(fibdir_theta, [length(qhat) 1])');
            fibdotgrad_phi = sum((qhat).*repmat(fibdir_phi, [length(qhat) 1])');
            sumResDerivtheta = S0*(-2*bvals*diff.*fibdotgrad_theta.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            sumResDerivphi = S0*(-2*bvals*diff.*fibdotgrad_phi.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            
            sumResDeriv = zeros(5, 1);
            sumResDeriv(1) = sum(-2*sumResDerivS0*(Avox - S'));
            sumResDeriv(2) = sum(-2*sumResDerivdiff*(Avox - S'));
            sumResDeriv(3) = sum(-2*sumResDerivf*(Avox - S'));
            sumResDeriv(4) = sum(-2*sumResDerivtheta*(Avox - S'));
            sumResDeriv(5) = sum(-2*sumResDerivphi*(Avox - S'));        
            %{
            sumResDeriv(1) = -((Avox - S').')*sumResDerivS0 - sumResDerivS0*(Avox - S');
            sumResDeriv(2) = -((Avox - S').')*sumResDerivdiff - sumResDerivdiff*(Avox - S');
            sumResDeriv(3) = -((Avox - S').')*sumResDerivf - sumResDerivf*(Avox - S');
            sumResDeriv(4) = -((Avox - S').')*sumResDerivtheta - sumResDerivtheta*(Avox - S');
            sumResDeriv(5) = -((Avox - S').')*sumResDerivphi - sumResDerivphi*(Avox - S');     
            %}
        end
    end
end

%}

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim2GRAD(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function [sumRes, sumResDeriv] = BallStickSSD(x)
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
        if nargout > 1 % gradient required
            sumResDerivS0 = f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff);
            sumResDerivdiff = S0*(-bvals.*(fibdotgrad.^2).*f.*exp(-bvals*diff.*(fibdotgrad.^2)) -bvals*(1-f).*exp(-bvals*diff));
            sumResDerivf = S0*(exp(-bvals*diff.*(fibdotgrad.^2)) - exp(-bvals*diff));

            %theta_inter = (cos(phi).^2)*2*sin(theta)*cos(theta).*(qhat(1, :).^2) + 2*(sin(phi).^2)*cos(theta)*sin(theta).*(qhat(2, :).^2) - 2*cos(theta)*sin(theta).*qhat(3, :) + 4*cos(phi)*sin(phi)*cos(theta)*sin(theta).*qhat(1, :).*qhat(2, :) + 2*cos(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(1, :).*qhat(3, :) + 2*sin(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(2, :).*qhat(3, :);
            %phi_inter = -2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(1, :).^2) + 2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(2, :).^2) + 2*((cos(phi).^2) - (sin(phi).^2))*(sin(theta).^2).*qhat(1, :).*qhat(2, :) -2*sin(phi)*sin(theta)*cos(theta).*qhat(1, :).*qhat(3, :) +2*cos(phi)*sin(theta)*cos(theta).*qhat(2, :).*qhat(3, :);
            %sumResDerivtheta = S0*(-f*theta_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));
            %sumResDerivphi = S0*(-f*phi_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));

            fibdir_theta = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
            fibdir_phi = [-sin(phi)*sin(theta) cos(phi)*sin(theta) 0.0];
            %fibdotgrad_theta = sum((qhat.^2).*repmat(fibdir_theta, [length(qhat) 1])');
            %fibdotgrad_phi = sum((qhat.^2).*repmat(fibdir_phi, [length(qhat) 1])');
            fibdotgrad_theta = sum((qhat).*repmat(fibdir_theta, [length(qhat) 1])');
            fibdotgrad_phi = sum((qhat).*repmat(fibdir_phi, [length(qhat) 1])');
            sumResDerivtheta = S0*(-2*bvals*diff.*fibdotgrad_theta.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            sumResDerivphi = S0*(-2*bvals*diff.*fibdotgrad_phi.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));

            sumResDeriv = zeros(5, 1);
            sumResDeriv(1) = sum(-2*sumResDerivS0*(Avox - S'));
            sumResDeriv(2) = sum(-2*sumResDerivdiff*(Avox - S'));
            sumResDeriv(3) = sum(-2*sumResDerivf*(Avox - S'));
            sumResDeriv(4) = sum(-2*sumResDerivtheta*(Avox - S'));
            sumResDeriv(5) = sum(-2*sumResDerivphi*(Avox - S'));  
        end
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = BallStickSSDOptim3GRAD(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function [sumRes, sumResDeriv] = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        %theta = (exp(-(x(4)^2))-0.5)*(pi/2);
        %phi = (exp(-(x(5)^2))-0.5)*pi;
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
        if nargout > 1 % gradient required
            sumResDerivS0 = f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff);
            sumResDerivdiff = S0*(-bvals.*(fibdotgrad.^2).*f.*exp(-bvals*diff.*(fibdotgrad.^2)) -bvals*(1-f).*exp(-bvals*diff));
            sumResDerivf = S0*(exp(-bvals*diff.*(fibdotgrad.^2)) - exp(-bvals*diff));

            %theta_inter = (cos(phi).^2)*2*sin(theta)*cos(theta).*(qhat(1, :).^2) + 2*(sin(phi).^2)*cos(theta)*sin(theta).*(qhat(2, :).^2) - 2*cos(theta)*sin(theta).*qhat(3, :) + 4*cos(phi)*sin(phi)*cos(theta)*sin(theta).*qhat(1, :).*qhat(2, :) + 2*cos(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(1, :).*qhat(3, :) + 2*sin(phi)*((cos(theta).^2) - (sin(theta).^2)).*qhat(2, :).*qhat(3, :);
            %phi_inter = -2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(1, :).^2) + 2*cos(phi)*sin(phi)*(sin(theta).^2).*(qhat(2, :).^2) + 2*((cos(phi).^2) - (sin(phi).^2))*(sin(theta).^2).*qhat(1, :).*qhat(2, :) -2*sin(phi)*sin(theta)*cos(theta).*qhat(1, :).*qhat(3, :) +2*cos(phi)*sin(theta)*cos(theta).*qhat(2, :).*qhat(3, :);
            %sumResDerivtheta = S0*(-f*theta_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));
            %sumResDerivphi = S0*(-f*phi_inter.*exp(-bvals*diff.*(fibdotgrad.^2)) - (1-f)*exp(-bvals*diff));

            fibdir_theta = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
            fibdir_phi = [-sin(phi)*sin(theta) cos(phi)*sin(theta) 0.0];
            %fibdotgrad_theta = sum((qhat.^2).*repmat(fibdir_theta, [length(qhat) 1])');
            %fibdotgrad_phi = sum((qhat.^2).*repmat(fibdir_phi, [length(qhat) 1])');
            fibdotgrad_theta = sum((qhat).*repmat(fibdir_theta, [length(qhat) 1])');
            fibdotgrad_phi = sum((qhat).*repmat(fibdir_phi, [length(qhat) 1])');
            sumResDerivtheta = S0*(-2*bvals*diff.*fibdotgrad_theta.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));
            sumResDerivphi = S0*(-2*bvals*diff.*fibdotgrad_phi.*fibdotgrad*f.*exp(-bvals*diff.*(fibdotgrad.^2)));

            sumResDeriv = zeros(5, 1);
            sumResDeriv(1) = sum(-2*sumResDerivS0*(Avox - S'));
            sumResDeriv(2) = sum(-2*sumResDerivdiff*(Avox - S'));
            sumResDeriv(3) = sum(-2*sumResDerivf*(Avox - S'));
            sumResDeriv(4) = sum(-2*sumResDerivtheta*(Avox - S'));
            sumResDeriv(5) = sum(-2*sumResDerivphi*(Avox - S'));  
        end
    end
end












