clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                       COMP0118 - Mini-project                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Task 1 – Load and plot the MRI volumes and masks
%% Loading using niftiread

pip0101_data = double(niftiread('data/pip0101_20_20_1401_T2MEdiff_abs.alle.nii'));
pip0101_mask = double(niftiread('data/pip0101_placenta_mask.nii'));
pip0101_full_mask = double(niftiread('data/pip0101_placenta_and_uterine_wall_mask.nii'));

pip0120_data = double(niftiread('data/pip0120_20_20_2501_T2MEdiff_abs.alle.nii'));
pip0120_mask = double(niftiread('data/pip0120_placenta_mask.nii'));
pip0120_full_mask = double(niftiread('data/pip0120_placenta_and_uterine_wall_mask.nii'));


%% Plotting some examples

figure();
subplot(2, 5, 1);
to_plot = pip0101_data(:, :, 1, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 2);
to_plot = pip0101_data(:, :, 5, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 3);
to_plot = pip0101_data(:, :, 10, 1);
imshow(to_plot/max(to_plot(:)));
title({'Plotting the data along some dimensions','Healthy control'});
xlabel('Plotting along the 3rd dimension');
subplot(2, 5, 4);
to_plot = pip0101_data(:, :, 20, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 5);
to_plot = pip0101_data(:, :, 30, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 6);
to_plot = pip0101_data(:, :, 15, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 7);
to_plot = pip0101_data(:, :, 15, 15);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 8);
to_plot = pip0101_data(:, :, 15, 50);
imshow(to_plot/max(to_plot(:)));
xlabel('Plotting along the 4th dimension');
subplot(2, 5, 9);
to_plot = pip0101_data(:, :, 15, 150);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 10);
to_plot = pip0101_data(:, :, 15, 300);
imshow(to_plot/max(to_plot(:)));

figure();
subplot(2, 5, 1);
to_plot = pip0120_data(:, :, 1, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 2);
to_plot = pip0120_data(:, :, 5, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 3);
to_plot = pip0120_data(:, :, 10, 1);
imshow(to_plot/max(to_plot(:)));
title({'Plotting the data along some dimensions','Pre-eclampsia patient'});
xlabel('Plotting along the 3rd dimension');
subplot(2, 5, 4);
to_plot = pip0120_data(:, :, 15, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 5);
to_plot = pip0120_data(:, :, 20, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 6);
to_plot = pip0120_data(:, :, 10, 1);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 7);
to_plot = pip0120_data(:, :, 10, 15);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 8);
to_plot = pip0120_data(:, :, 10, 50);
imshow(to_plot/max(to_plot(:)));
xlabel('Plotting along the 4th dimension');
subplot(2, 5, 9);
to_plot = pip0120_data(:, :, 10, 150);
imshow(to_plot/max(to_plot(:)));
subplot(2, 5, 10);
to_plot = pip0120_data(:, :, 10, 300);
imshow(to_plot/max(to_plot(:)));

figure();
sgtitle({'ROI mask of the uterine wall+placenta', 'Healthy control'});
for kk=1:30
    subplot(3, 10, kk);
    imshow(pip0101_full_mask(:, :, kk));
end

figure();
sgtitle({'ROI mask of the uterine wall+placenta', 'Pre-eclampsia participant'});
for kk=1:20
    subplot(2, 10, kk);
    imshow(pip0101_full_mask(:, :, kk));
end


%% Plotting some examples (one-by-one)

%{
figure();
fig_to_plot = flipud(pip0101_data(:, :, 20, 1)');
imshow(fig_to_plot/max(fig_to_plot(:)));
%}


%% Task 2 – Load and plot the MRI acquisition parameters
%% Loading the text file

gradecho = importdata('data/grad_echo.txt');

% The 4th dimension in the data correspond to the index row in the gradecho
% file
% Unique b-values: 14 (0, 5, 10, 18, 25, 36, 50, 100, 200, 400, 600, 800,
% 1200, 1600)
% Unique echo times: 5 (78, 114, 150, 186, 222)


%% Visualizing b-values VS echo times

figure();
scatter(gradecho(:, 5), gradecho(:, 4));
xlabel('Echo time (s)');
ylabel('b-value (s/mm^2)')


%% Task 3 – Fit the combined T2*-ADC model
%% Importing slice 7

healthy_07 = squeeze(pip0101_data(:, :, 7, :));
preecl_07 = squeeze(pip0120_data(:, :, 7, :));

mask_healthy_07 = squeeze(pip0101_full_mask(:, :, 7));
mask_preecl_07 = squeeze(pip0120_full_mask(:, :, 7));
mask_placenta_healthy_07 = squeeze(pip0101_mask(:, :, 7));
mask_placenta_preecl_07 = squeeze(pip0120_mask(:, :, 7));
% create a temporary mask
% fitting: full mask (generally better, depends)


%% Using a linear model
% Design matrix

G = [-gradecho(:, 5) -gradecho(:, 4)];


%% Using a linear model
% Iterating for the healthy model

% Storage for our answers
res_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 2);
res_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), 2);

tic;
% Iterating over all the voxels
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % Creating the A matrix
            A_ii_jj = log(squeeze(healthy_07(ii, jj, :))) - log(squeeze(healthy_07(ii, jj, 1)));

            % Computing x using pinv
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)

            % Storing
            res_healthy(ii, jj, :) = x_ii_jj;

        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % Creating the A matrix
            A_ii_jj = log(squeeze(preecl_07(ii, jj, :))) - log(squeeze(preecl_07(ii, jj, 1)));

            % Computing x using pinv
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)

            % Storing
            res_preecl(ii, jj, :) = x_ii_jj;

        end
    end
end

toc;

%% Creating the data obtained with the linear model

S_est_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), size(healthy_07, 3));
S_est_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), size(preecl_07, 3));
RESNORM_ADC_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
RESNORM_ADC_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));

for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            S_est_healthy(ii, jj, :) = squeeze(healthy_07(ii, jj, 1)) * exp(-gradecho(:, 5)*res_healthy(ii, jj, 1)) .* exp(-gradecho(:, 4)*res_healthy(ii, jj, 2));
            RESNORM_ADC_07_healthy(ii, jj) = ADC_SSD(squeeze(healthy_07(ii, jj, :)), squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), 1/res_healthy(ii, jj, 1), gradecho(:, 4), res_healthy(ii , jj, 2));

        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            S_est_preecl(ii, jj, :) = squeeze(preecl_07(ii, jj, 1)) * exp(-gradecho(:, 5)*res_preecl(ii, jj, 1)) .* exp(-gradecho(:, 4)*res_preecl(ii, jj, 2));
            RESNORM_ADC_07_preecl(ii, jj) = ADC_SSD(squeeze(preecl_07(ii, jj, :)), squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), 1/res_preecl(ii, jj, 1), gradecho(:, 4), res_preecl(ii , jj, 2));

        end
    end
end

%% Task 4 – Compare maps for control and pre-eclampsia participants
%% Visualizing (plot)

figure();
plot(squeeze(pip0101_data(100, 45, 7, :)), '+');
hold on;
plot(squeeze(S_est_healthy(100, 45, :)), '.');
hold off;
title({'Comparison data VS estimation for the healthy control'});
legend('Data', 'Estimation');

figure();
plot(squeeze(pip0120_data(100, 45, 7, :)), '+');
hold on;
plot(squeeze(S_est_preecl(100, 45, :)), '.');
hold off;
title('Comparison data VS estimation for the pre-eclampsia patient');
legend('Data', 'Estimation');

% Not that great of a fit.. Could be useful for starting points

%% Maps for T2* and D
% Beware: x contains (1/T2* D)
% colormap command: set(0,'DefaultFigureColormap',feval('hot'));
% Required add-on: https://www.mathworks.com/matlabcentral/fileexchange/42904-imoverlay
% (was renamed to imoverlay2 locally because imoverlay already exists in
% the Image processing toolbox)

% Healthy T2*
to_plot_bg = squeeze(pip0101_data(:, :, 7, 1));
to_plot_bg = to_plot_bg/max(to_plot_bg(:));
to_plot_fg = res_healthy(:, :, 1);
to_plot_fg = 1./to_plot_fg;
to_plot_fg(isinf(to_plot_fg)) = 0;
to_plot_fg = to_plot_fg/max(to_plot_fg(:));
[hF, hB] = imoverlay2(to_plot_bg, to_plot_fg, [], [], colormap);
hold on;
colorbar;
hold off;

% Healthy D
to_plot_bg = squeeze(pip0101_data(:, :, 7, 1));
to_plot_bg = to_plot_bg/max(to_plot_bg(:));
to_plot_fg = res_healthy(:, :, 2);
to_plot_fg = to_plot_fg/max(to_plot_fg(:));
[hF, hB] = imoverlay2(to_plot_bg, to_plot_fg, [], [], colormap);
hold on;
colorbar;
hold off;

% Pre-eclampsia T2*
to_plot_bg = squeeze(pip0120_data(:, :, 7, 1));
to_plot_bg = to_plot_bg/max(to_plot_bg(:));
to_plot_fg = res_preecl(:, :, 1);
to_plot_fg(to_plot_fg<0) = 0; % Numerical error: we have a negative value
to_plot_fg = 1./to_plot_fg;
to_plot_fg(isinf(to_plot_fg)) = 0;
to_plot_fg = to_plot_fg/max(to_plot_fg(:));
to_plot_fg(to_plot_fg==1) = 0;  % Better-looking mapping
to_plot_fg = to_plot_fg/max(to_plot_fg(:));
[hF, hB] = imoverlay2(to_plot_bg, to_plot_fg, [], [], colormap);
hold on;
colorbar;
hold off;

% Pre-eclampsia D
to_plot_bg = squeeze(pip0120_data(:, :, 7, 1));
to_plot_bg = to_plot_bg/max(to_plot_bg(:));
to_plot_fg = res_preecl(:, :, 2);
to_plot_fg = to_plot_fg/max(to_plot_fg(:));
[hF, hB] = imoverlay2(to_plot_bg, to_plot_fg, [], [], colormap);
hold on;
colorbar;
hold off;


%% Task 5 – Fit the combined T2*-Intravoxel incoherent motion (IVIM) model
%% Optimization routine

% Storage
T2_star_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
D_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
Dp_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
f_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
RESNORM_IVIM_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
T2_star_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
D_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
Dp_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
f_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
RESNORM_IVIM_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));

% D/Dp ratio
Dmax = 1e-2;
Dx = 10;

% fmincon parameters
h=optimset('MaxFunEvals',10000,'Algorithm','interior-point','TolX',1e-8,'TolFun',1e-8,'display','off');
%h=optimset('MaxFunEvals',5000,'Algorithm','interior-point','TolX',1e-3,'TolFun',1e-3);
% fminunc parameters

%test_resnorm = zeros(1, 100);
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));
    
            % First estimation: linear model
            A_ii_jj = log(voxOfInterest) - log(squeeze(healthy_07(ii, jj, 1)));
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)
            
            % Start-point estimate
            startx = [1/x_ii_jj(1) x_ii_jj(2) x_ii_jj(2)*Dx 0.5];
            % Adding some noise
            startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];

            % Optimization
            [parameter_hat,RESNORM,~,~]=IVIM_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:5
                startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [parameter_hat,RESNORM,~,~]=IVIM_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end
            end

            % Storing
            T2_star_07_healthy(ii, jj) = best_parameter_hat(1);
            D_07_healthy(ii, jj) = best_parameter_hat(2);
            Dp_07_healthy(ii, jj) = best_parameter_hat(3);
            f_07_healthy(ii, jj) = best_parameter_hat(4);
            RESNORM_IVIM_07_healthy(ii, jj) = best_RESNORM;

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));
    
            % First estimation: linear model
            A_ii_jj = log(voxOfInterest) - log(squeeze(preecl_07(ii, jj, 1)));
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)
            
            % Start-point estimate
            startx = [1/x_ii_jj(1) x_ii_jj(2) x_ii_jj(2)*Dx 0.5];
            % Adding some noise
            startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];

            % Optimization
            [parameter_hat,RESNORM,~,~]=IVIM_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:5
                startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                [parameter_hat,RESNORM,~,~]=IVIM_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end
            end

            % Storing
            T2_star_07_preecl(ii, jj) = best_parameter_hat(1);
            D_07_preecl(ii, jj) = best_parameter_hat(2);
            Dp_07_preecl(ii, jj) = best_parameter_hat(3);
            f_07_preecl(ii, jj) = best_parameter_hat(4);
            RESNORM_IVIM_07_preecl(ii, jj) = best_RESNORM;

        end
    end
end
toc;


%% Colormap

cmap = colormap('hot');


%% Visualizing our results

figure();
subplot(181);
to_plot = res_healthy(:, :, 1);
to_plot = 1./to_plot;
to_plot(isinf(to_plot)) = 0;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap)
title("T_{2}^{*} (ADC model)");
xlabel('Healthy control');
subplot(182);
to_plot = T2_star_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("T_{2}^{*} (IVIM model)");
xlabel('Healthy control');
subplot(183);
to_plot = res_healthy(:, :, 2);
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("D (ADC model)");
xlabel('Healthy control');
subplot(184);
to_plot = D_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("D (IVIM model)");
xlabel('Healthy control');
subplot(185);
to_plot = Dp_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("D_p (IVIM model)");
xlabel('Healthy control');
subplot(186);
to_plot = T2_star_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("T_{2}^{*} (IVIM model)");
xlabel('Pre-eclampsia patient');
subplot(187);
to_plot = D_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("D (IVIM model)");
xlabel('Pre-eclampsia patient');
subplot(188);
to_plot = Dp_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title("D_p (IVIM model)");
xlabel('Pre-eclampsia patient');


figure();
subplot(161);
to_plot = T2_star_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('T2^{*} with the optimization')
subplot(162);
to_plot = res_healthy(:, :, 1);
to_plot = 1./to_plot;
to_plot(isinf(to_plot)) = 0;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap)
title('T2^{*} with the linear model')
subplot(163);
to_plot = Dp_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D_{p} with the optimization');
subplot(164);
to_plot = D_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D with the optimization')
subplot(165);
to_plot = res_healthy(:, :, 2);
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D with the linear model');
subplot(166);
to_plot = f_07_healthy;
imshow(to_plot, Colormap=cmap);
title('f with the optimization');
sgtitle('Optimized parameters for the healthy control');


figure();
subplot(161);
to_plot = T2_star_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('T2^{*} with the optimization')
subplot(162);
to_plot = res_preecl(:, :, 1);
to_plot = 1./to_plot;
to_plot(isinf(to_plot)) = 0;
to_plot(to_plot>0.2) = 0; %abnormal values
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap)
title('T2^{*} with the linear model')
subplot(163);
to_plot = Dp_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D_{p} with the optimization');
subplot(164);
to_plot = D_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D with the optimization')
subplot(165);
to_plot = res_preecl(:, :, 2);
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('D with the linear model');
subplot(166);
to_plot = f_07_preecl;
imshow(to_plot, Colormap=cmap);
title('f with the optimization');
sgtitle('Optimized parameters for the pre-eclampsia patient');


%% Boxplots


figure();
subplot(121);
boxplot(reshape(T2_star_07_healthy, [], 1));
ylim([0.02 0.28]);
ylabel('T_2^* (s)')
title('Healthy control');
subplot(122);
boxplot(reshape(T2_star_07_preecl, [], 1));
ylim([0.02 0.28]);
title('Pre-eclampsia patient')


%% TASK 6
%% Computing the BIC

BIC_ADC_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
BIC_IVIM_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
BIC_ADC_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));
BIC_IVIM_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));

AIC_ADC_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
AIC_IVIM_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
AIC_ADC_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));
AIC_IVIM_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));

K = length(gradecho(:, 4));

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Checking the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % Computing the BIC
            % K=length(gradecho(4,:)) for both, N=5 for ADC, N=7
            %BIC_ADC_healthy_07(ii, jj) = 5 * log(K) - 2 * log(ADC_SSD(squeeze(healthy_07(ii, jj, :)), squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), 1/res_healthy(ii, jj, 1), gradecho(:, 4), res_healthy(ii, jj, 2)));
            %BIC_IVIM_healthy_07(ii, jj) = 7 * log(K) - 2 * log(IVIM_SSD(squeeze(healthy_07(ii, jj, :)), squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), T2_star_07_healthy(ii, jj), gradecho(:, 4), D_07_healthy(ii, jj), Dp_07_healthy(ii, jj), f_07_healthy(ii, jj)));
            BIC_ADC_healthy_07(ii, jj) = 5 * log(K) + K * log((1/K)*RESNORM_ADC_07_healthy(ii, jj));
            BIC_IVIM_healthy_07(ii, jj) = 7 * log(K) + K * log((1/K)*RESNORM_IVIM_07_healthy(ii, jj));
            AIC_ADC_healthy_07(ii, jj) = 2 * 5 + K * log((1/K)*RESNORM_ADC_07_healthy(ii, jj));
            AIC_IVIM_healthy_07(ii, jj) = 2 * 7 + K * log((1/K)*RESNORM_IVIM_07_healthy(ii, jj));

        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Checking the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % Computing the BIC
            %BIC_ADC_preecl_07(ii, jj) = 5 * log(K) - 2 * log(ADC_SSD(squeeze(preecl_07(ii, jj, :)), squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), 1/res_preecl(ii, jj, 1), gradecho(:, 4), res_preecl(ii, jj, 2)));
            %BIC_IVIM_preecl_07(ii, jj) = 7 * log(K) - 2 * log(IVIM_SSD(squeeze(preecl_07(ii, jj, :)), squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), T2_star_07_preecl(ii, jj), gradecho(:, 4), D_07_preecl(ii, jj), Dp_07_preecl(ii, jj), f_07_preecl(ii, jj)));
            BIC_ADC_preecl_07(ii, jj) = 5 * log(K) + K * log((1/K)*RESNORM_ADC_07_preecl(ii, jj));
            BIC_IVIM_preecl_07(ii, jj) = 7 * log(K) + K * log((1/K)*RESNORM_IVIM_07_preecl(ii, jj));
            AIC_ADC_preecl_07(ii, jj) = 2 * 5 + K * log((1/K)*RESNORM_ADC_07_preecl(ii, jj));
            AIC_IVIM_preecl_07(ii, jj) = 2 * 7 + K * log((1/K)*RESNORM_IVIM_07_preecl(ii, jj));

        end
    end
end
toc;


%% Computing the map showing the best BIC per voxel

% Initializing the maps
BIC_map_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 3);
BIC_map_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), 3);
AIC_map_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 3);
AIC_map_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), 3);

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Checking the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % Putting a color depending on which BIC is best (lowest)
            % Red => ADC; Blue => IVIM
            if(BIC_ADC_healthy_07(ii, jj) < BIC_IVIM_healthy_07(ii, jj))
                BIC_map_healthy(ii, jj, :) = [1 0 0];
            else
                BIC_map_healthy(ii, jj, :) = [0 0 1];
            end
            if(AIC_ADC_healthy_07(ii, jj) < AIC_IVIM_healthy_07(ii, jj))
                AIC_map_healthy(ii, jj, :) = [1 0 0];
            else
                AIC_map_healthy(ii, jj, :) = [0 0 1];
            end
        
        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Checking the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % Putting a color depending on which BIC is best (lowest)
            % Red => ADC; Blue => IVIM
            if(BIC_ADC_preecl_07(ii, jj) < BIC_IVIM_preecl_07(ii, jj))
                BIC_map_preecl(ii, jj, :) = [1 0 0];
            else
                BIC_map_preecl(ii, jj, :) = [0 0 1];
            end
            if(AIC_ADC_preecl_07(ii, jj) < AIC_IVIM_preecl_07(ii, jj))
                AIC_map_preecl(ii, jj, :) = [1 0 0];
            else
                AIC_map_preecl(ii, jj, :) = [0 0 1];
            end

        end
    end
end
toc;


%% Visualizing the maps

figure();
subplot(121)
imshow(BIC_map_healthy);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Healthy control');
subplot(122)
imshow(BIC_map_preecl);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Pre-eclampsia patient');
sgtitle('BIC map (the color represents the best model i.e lowest BIC)')

figure();
subplot(121)
imshow(AIC_map_healthy);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Healthy control');
subplot(122)
imshow(AIC_map_preecl);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Pre-eclampsia patient');
sgtitle('AIC map (the color represents the best model i.e lowest AIC)')


%% Task 5 bis – Fit the combined T2*-Intravoxel incoherent motion (IVIM) model w/ fminunc
%% Optimization routine

% Storage
T2_star_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
D_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
Dp_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
f_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
RESNORM_IVIM_07_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
T2_star_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
D_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
Dp_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
f_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));
RESNORM_IVIM_07_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));

% D/Dp ratio
Dmax = 1e-2;
Dx = 10;

% fminunc parameters
h=optimset('MaxFunEvals',20000,'Algorithm','quasi-newton','TolX',1e-9,'TolFun',1e-9,'display','off');

test_resnorm = zeros(1, 100);
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));
    
            % First estimation: linear model
            A_ii_jj = log(voxOfInterest) - log(squeeze(healthy_07(ii, jj, 1)));
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)
            
            % Start-point estimate
            startx = [real(sqrt(-log((1/x_ii_jj(1))/0.25))) real(sqrt(-log(x_ii_jj(2)/(Dmax/Dx)))) real(sqrt(-log(x_ii_jj(2)/(Dmax - (Dmax/Dx))))-(Dmax/Dx)) real(sqrt(-log(0.5)))];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                    [parameter_hat,RESNORM,~,~]=IVIM_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:5
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                        [parameter_hat,RESNORM,~,~]=IVIM_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            T2_star_07_healthy(ii, jj) = exp(-(best_parameter_hat(1).^2))*0.25;
            D_07_healthy(ii, jj) = exp(-(best_parameter_hat(2).^2))*(Dmax/Dx);
            Dp_07_healthy(ii, jj) = exp(-(best_parameter_hat(3).^2))*(Dmax - (Dmax/Dx)) + (Dmax/Dx);
            f_07_healthy(ii, jj) = exp(-(best_parameter_hat(4).^2));
            RESNORM_IVIM_07_healthy(ii, jj) = best_RESNORM;

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));
    
            % First estimation: linear model
            A_ii_jj = log(voxOfInterest) - log(squeeze(preecl_07(ii, jj, 1)));
            x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)
            
            % Start-point estimate
            startx = [real(sqrt(-log((1/x_ii_jj(1))/0.25))) real(sqrt(-log(x_ii_jj(2)/(Dmax/Dx)))) real(sqrt(-log(x_ii_jj(2)/(Dmax - (Dmax/Dx))))-(Dmax/Dx)) real(sqrt(-log(0.5)))];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                    [parameter_hat,RESNORM,~,~]=IVIM_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:5

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5) normrnd(0, 0.5)];
                        [parameter_hat,RESNORM,~,~]=IVIM_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), gradecho(:, 4), Dmax, Dx, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            T2_star_07_preecl(ii, jj) = exp(-(best_parameter_hat(1).^2))*0.25;
            D_07_preecl(ii, jj) = exp(-(best_parameter_hat(2).^2))*(Dmax/Dx);
            Dp_07_preecl(ii, jj) = exp(-(best_parameter_hat(3).^2))*(Dmax - (Dmax/Dx)) + (Dmax/Dx);
            f_07_preecl(ii, jj) = exp(-(best_parameter_hat(4).^2));
            RESNORM_IVIM_07_preecl(ii, jj) = best_RESNORM;

        end
    end
end
toc;


%% Visualizing our results

figure();
subplot(161);
to_plot = T2_star_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('T2^{*} with the optimization')
subplot(162);
to_plot = res_healthy(:, :, 1);
to_plot = 1./to_plot;
to_plot(isinf(to_plot)) = 0;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot)
title('T2^{*} with the linear model')
subplot(163);
to_plot = Dp_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D_{p} with the optimization');
subplot(164);
to_plot = D_07_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D with the optimization')
subplot(165);
to_plot = res_healthy(:, :, 2);
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D with the linear model');
subplot(166);
to_plot = f_07_healthy;
imshow(to_plot);
title('f with the optimization');
sgtitle('Optimized parameters for the healthy control');

figure();
subplot(161);
to_plot = T2_star_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('T2^{*} with the optimization')
subplot(162);
to_plot = res_preecl(:, :, 1);
to_plot = 1./to_plot;
to_plot(isinf(to_plot)) = 0;
to_plot(to_plot>0.2) = 0; %abnormal values
to_plot = to_plot/max(to_plot(:));
imshow(to_plot)
title('T2^{*} with the linear model')
subplot(163);
to_plot = Dp_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D_{p} with the optimization');
subplot(164);
to_plot = D_07_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D with the optimization')
subplot(165);
to_plot = res_preecl(:, :, 2);
to_plot = to_plot/max(to_plot(:));
imshow(to_plot);
title('D with the linear model');
subplot(166);
to_plot = f_07_preecl;
imshow(to_plot);
title('f with the optimization');
sgtitle('Optimized parameters for the pre-eclampsia patient');


%% TASK 6 bis
%% Computing the BIC

BIC_ADC_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
BIC_IVIM_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
BIC_ADC_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));
BIC_IVIM_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));

AIC_ADC_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
AIC_IVIM_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2));
AIC_ADC_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));
AIC_IVIM_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2));

K = length(gradecho(:, 4));

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Checking the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % Computing the BIC
            % K=length(gradecho(4,:)) for both, N=5 for ADC, N=7
            %BIC_ADC_healthy_07(ii, jj) = 5 * log(K) - 2 * log(ADC_SSD(squeeze(healthy_07(ii, jj, :)), squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), 1/res_healthy(ii, jj, 1), gradecho(:, 4), res_healthy(ii, jj, 2)));
            %BIC_IVIM_healthy_07(ii, jj) = 7 * log(K) - 2 * log(IVIM_SSD(squeeze(healthy_07(ii, jj, :)), squeeze(healthy_07(ii, jj, 1)), gradecho(:, 5), T2_star_07_healthy(ii, jj), gradecho(:, 4), D_07_healthy(ii, jj), Dp_07_healthy(ii, jj), f_07_healthy(ii, jj)));
            BIC_ADC_healthy_07(ii, jj) = 5 * log(K) + K * log((1/K)*RESNORM_ADC_07_healthy(ii, jj));
            BIC_IVIM_healthy_07(ii, jj) = 7 * log(K) + K * log((1/K)*RESNORM_IVIM_07_healthy(ii, jj));
            AIC_ADC_healthy_07(ii, jj) = 2 * 5 + K * log((1/K)*RESNORM_ADC_07_healthy(ii, jj));
            AIC_IVIM_healthy_07(ii, jj) = 2 * 7 + K * log((1/K)*RESNORM_IVIM_07_healthy(ii, jj));

        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Checking the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % Computing the BIC
            %BIC_ADC_preecl_07(ii, jj) = 5 * log(K) - 2 * log(ADC_SSD(squeeze(preecl_07(ii, jj, :)), squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), 1/res_preecl(ii, jj, 1), gradecho(:, 4), res_preecl(ii, jj, 2)));
            %BIC_IVIM_preecl_07(ii, jj) = 7 * log(K) - 2 * log(IVIM_SSD(squeeze(preecl_07(ii, jj, :)), squeeze(preecl_07(ii, jj, 1)), gradecho(:, 5), T2_star_07_preecl(ii, jj), gradecho(:, 4), D_07_preecl(ii, jj), Dp_07_preecl(ii, jj), f_07_preecl(ii, jj)));
            BIC_ADC_preecl_07(ii, jj) = 5 * log(K) + K * log((1/K)*RESNORM_ADC_07_preecl(ii, jj));
            BIC_IVIM_preecl_07(ii, jj) = 7 * log(K) + K * log((1/K)*RESNORM_IVIM_07_preecl(ii, jj));
            AIC_ADC_preecl_07(ii, jj) = 2 * 5 + K * log((1/K)*RESNORM_ADC_07_preecl(ii, jj));
            AIC_IVIM_preecl_07(ii, jj) = 2 * 7 + K * log((1/K)*RESNORM_IVIM_07_preecl(ii, jj));

        end
    end
end
toc;


%% Computing the map showing the best BIC per voxel

% Initializing the maps
BIC_map_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 3);
BIC_map_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), 3);
AIC_map_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 3);
AIC_map_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2), 3);

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Checking the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % Putting a color depending on which BIC is best (lowest)
            % Red => ADC; Blue => IVIM
            if(BIC_ADC_healthy_07(ii, jj) < BIC_IVIM_healthy_07(ii, jj))
                BIC_map_healthy(ii, jj, :) = [1 0 0];
            else
                BIC_map_healthy(ii, jj, :) = [0 0 1];
            end
            if(AIC_ADC_healthy_07(ii, jj) < AIC_IVIM_healthy_07(ii, jj))
                AIC_map_healthy(ii, jj, :) = [1 0 0];
            else
                AIC_map_healthy(ii, jj, :) = [0 0 1];
            end
        
        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Checking the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % Putting a color depending on which BIC is best (lowest)
            % Red => ADC; Blue => IVIM
            if(BIC_ADC_preecl_07(ii, jj) < BIC_IVIM_preecl_07(ii, jj))
                BIC_map_preecl(ii, jj, :) = [1 0 0];
            else
                BIC_map_preecl(ii, jj, :) = [0 0 1];
            end
            if(AIC_ADC_preecl_07(ii, jj) < AIC_IVIM_preecl_07(ii, jj))
                AIC_map_preecl(ii, jj, :) = [1 0 0];
            else
                AIC_map_preecl(ii, jj, :) = [0 0 1];
            end

        end
    end
end
toc;


%% Visualizing the maps

figure();
subplot(121)
imshow(BIC_map_healthy);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Healthy control');
subplot(122)
imshow(BIC_map_preecl);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Pre-eclampsia patient');
sgtitle('BIC map (the color represents the best model i.e lowest BIC)')

figure();
subplot(121)
imshow(AIC_map_healthy);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Healthy control');
subplot(122)
imshow(AIC_map_preecl);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('ADC', 'IVIM');
title('Pre-eclampsia patient');
sgtitle('AIC map (the color represents the best model i.e lowest AIC)')


%% TASK 7
%% Creating the data vectors

% Let's start with S (average over all spatial dimensions for each (b, TE))
N_S = 330;

% Healthy patient
S_healthy = zeros(N_S, 1);
S_healthy_only_placenta = zeros(N_S, 1);
pip0101_data_NaN = pip0101_data;
pip0101_data_only_placenta_NaN = pip0101_data;

% Iterating over all the (b, TE) measurements
tic;
for kk=1:N_S

    % We will replace the 0 values by NaN

    % Iterating over all voxels for the kk-th measurement
    for ii=1:200
        for jj=1:106
            for zz=30
                % Needs to be in the ROI
                % Uterine wall + placenta
                if(pip0101_full_mask(ii, jj, zz)==0)
                    pip0101_data_NaN(ii, jj, zz, kk) = NaN;
                end
                % Only placenta
                if(pip0101_mask(ii, jj, zz)==0)
                    pip0101_data_only_placenta_NaN(ii, jj, zz, kk) = NaN;
                end
            end
        end
    end

    % Storing the mean of it
    S_healthy(kk, 1) = mean(mean(mean(pip0101_data_NaN(:, :, :, kk), "omitnan"), "omitnan"), "omitnan");
    S_healthy_only_placenta(kk, 1) =  mean(mean(mean(pip0101_data_only_placenta_NaN(:, :, :, kk), "omitnan"), "omitnan"), "omitnan");
end
toc;


% Pre-eclampsia patient
S_preecl = zeros(N_S, 1);
S_preecl_only_placenta = zeros(N_S, 1);
pip0120_data_NaN = pip0120_data;
pip0120_data_only_placenta_NaN = pip0120_data;

% Iterating over all the (b, TE) measurements
tic;
for kk=1:N_S

    % We will replace the 0 values by NaN

    % Iterating over all voxels for the kk-th measurement
    for ii=1:200
        for jj=1:102
            for zz=20
                % Needs to be in the ROI
                % Uterine wall + placenta
                if(pip0120_full_mask(ii, jj, zz)==0)
                    pip0120_data_NaN(ii, jj, zz, kk) = NaN;
                end
                % Only placenta
                if(pip0120_mask(ii, jj, zz)==0)
                    pip0120_data_only_placenta_NaN(ii, jj, zz, kk) = NaN;
                end
            end
        end
    end

    % Storing the mean of it
    S_preecl(kk, 1) = mean(mean(mean(pip0120_data_NaN(:, :, :, kk), "omitnan"), "omitnan"), "omitnan");
    S_preecl_only_placenta(kk, 1) =  mean(mean(mean(pip0120_data_only_placenta_NaN(:, :, :, kk), "omitnan"), "omitnan"), "omitnan");
end
toc;


%% Discretization of T2* and D values - 1/2
% First we need to get an idea of the range of T2* and D values
% We will take a look at the ADC estimates

res_T2_inv_healthy = 1./res_healthy(:, :, 1);
res_T2_inv_healthy(isinf(res_T2_inv_healthy)) = NaN;
res_T2_inv_preecl = 1./res_preecl(:, :, 1);
res_T2_inv_preecl(isinf(res_T2_inv_preecl)) = NaN;
res_T2_inv_preecl(res_T2_inv_preecl<0) = NaN;

res_D_healthy = res_healthy(:, :, 2);
res_D_healthy(res_T2_inv_healthy==0) = NaN;
res_D_preecl = res_preecl(:, :, 2);
res_D_preecl(res_D_preecl==0) = NaN;
res_D_preecl(res_D_preecl<0) = NaN;

min_T2_star_healthy = min(min(res_T2_inv_healthy, [], "omitnan"), [], "omitnan");
max_T2_star_healthy = max(max(res_T2_inv_healthy, [], "omitnan"), [], "omitnan");
min_T2_star_preecl = min(min(res_T2_inv_preecl, [], "omitnan"), [], "omitnan");
max_T2_star_preecl = max(min(res_T2_inv_preecl, [], "omitnan"), [], "omitnan");
min_D_healthy = min(min(res_D_healthy, [], "omitnan"), [], "omitnan");
max_D_healthy = max(max(res_D_healthy, [], "omitnan"), [], "omitnan");
min_D_preecl = min(min(res_D_preecl, [], "omitnan"), [], "omitnan");
max_D_preecl = max(max(res_D_preecl, [], "omitnan"), [], "omitnan");


%% Discretization of T2* and D values - 2/2
% Secondly we create a grid of T2* and D values
% And we iterate over every (b, TE) values
    
% Creating the grids 
N_T2 = 50;
N_D = 30;

% Different grids
T2_star_healthy_grid = linspace(0.035, 0.130, N_T2);
D_healthy_grid = linspace(0, 0.0025, N_D);
T2_star_preecl_grid = linspace(0.042, 0.066, N_T2);
D_preecl_grid = linspace(0, 0.0026, N_D);

% Same grid
%T2_star_grid = linspace(0.035, 0.130, N_T2);
%D_grid = linspace(0, 0.0026, N_D);
T2_star_grid = linspace(0, 0.1, N_T2);
%T2_star_grid = linspace(0, 0.12, N_T2);
D_grid = linspace(0, 0.01, N_D);
%D_grid = logspace(-1, 3.5, N_D);


% Storage
K = zeros(N_S, N_T2*N_D);

% Iterating
for kk=1:N_S

    %{
    K_kk = zeros(N_T2, N_D);
    b_kk = gradecho(kk, 4);
    TE_kk = gradecho(kk, 5);

    for ii=1:N_T2
        for jj=1:N_D
            K_kk(ii, jj) = exp(-TE_kk/T2_star_grid(ii)) * exp(-b_kk*D_grid(jj));
        end
    end
    %}
    K_kk = zeros(N_D, N_T2);
    b_kk = gradecho(kk, 4);
    TE_kk = gradecho(kk, 5);

    for ii=1:N_D
        for jj=1:N_T2
            K_kk(ii, jj) = exp(-TE_kk/T2_star_grid(jj)) * exp(-b_kk*D_grid(ii));
        end
    end

    K(kk, :) = reshape(K_kk, 1, []);
end


%% Using the non-negative lsqr solver to compute the spectrum

h = optimset('Display','notify','TolX',1e-20);
F_healthy = reshape(lsqnonneg(K, S_healthy, h), N_T2, N_D);
F_preecl = reshape(lsqnonneg(K, S_preecl, h), N_T2, N_D);
F_healthy_only_placenta = reshape(lsqnonneg(K, S_healthy_only_placenta, h), N_T2, N_D);
F_preecl_only_placenta = reshape(lsqnonneg(K, S_preecl_only_placenta, h), N_T2, N_D);


%% Plotting

figure();
subplot(121);
contour(F_healthy);
title('Healthy subject');
xlabel('T2* (x 2ms)');
ylabel('ADC (x 0.33e-3 mm^{2}s^{-1})');
xlim([0 6]);
subplot(122);
contour(F_preecl);
title('Pre-eclampsia subject');
xlabel('T2* (x 2ms)');
ylabel('ADC (x 0.33e-3 mm^{2}s^{-1})');
xlim([0 6]);
sgtitle('Contour map of the T2*-D spectra');


%% Task 8 - Trying extended compartment models
%% Experimenting our models

% D/Dp ratio
Dmax = 1e-2;
Dx = 10;

% fminunc parameters
h_unc=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
h_con=optimset('MaxFunEvals',6500,'Algorithm','interior-point','TolX',1e-6,'TolFun',1e-6,'display','off');

%ii = 100;
%jj = 50;
nb_of_tries = 20;
nb_of_voxels = 5;
res_tries_unc = zeros(nb_of_voxels, nb_of_tries);
res_tries_con = zeros(nb_of_voxels, nb_of_tries);


for testtt=1:nb_of_voxels

ii=randi(200);
jj=randi(106);

% Verifying if we're in the ROI
while(mask_healthy_07(ii, jj) == 0)
    ii=randi(200);
    jj=randi(106);
end

    % We are working on this one
    voxOfInterest = squeeze(healthy_07(ii, jj, :));

    % First estimation: linear model
    A_ii_jj = log(voxOfInterest) - log(squeeze(healthy_07(ii, jj, 1)));
    x_ii_jj = pinv(G) * A_ii_jj; % Estimating (1/T2* D)

    start_D_unc = real(sqrt(-log(x_ii_jj(2))));
    start_D_con = x_ii_jj(2);
    start_T2_unc = real(sqrt(-log((1/x_ii_jj(1))/0.25)));
    start_T2_con = 1/x_ii_jj(1);

    % Start-point estimate
    %startx = [start_T2_con 600e-03 3e-03 0.45];
    %startx = zeros(1, 4);
    startx = [0.2 0.1 0.1e-03 0 0 0.1e-03 0.1e-03 0 0 0.5];

    % Optimization
    for kk=1:nb_of_tries
        startx = startx + [normrnd(0, 0.2) normrnd(0, 0.1) normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03) normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.2)];
        [~,RESNORM,~,~]=Zeppelin_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h_unc);
        res_tries_unc(testtt, kk) = RESNORM;
        [~,RESNORM,~,~]=Zeppelin_Zeppelin_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h_con);
        res_tries_con(testtt, kk) = RESNORM;
    end

end

figure();
subplot(121);
for ll=1:nb_of_voxels
    plot(res_tries_unc(ll, :));
    hold on;
end
hold off;
title('RESNORM over different fminunc runs');
xlabel('Run number');
ylabel('RESNORM');
subplot(122)
for ll=1:nb_of_voxels
    plot(res_tries_con(ll, :));
    hold on;
end
hold off;
title('RESNORM over different fmincon runs');
xlabel('Run number');
ylabel('RESNORM');

sgtitle('Zeppelin Zeppelin');


%% BALL BALL FITTING

% Storage
nb_of_params = 4;
param_07_healthy_ball_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_ball_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_ball_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_ball_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');

disp('Ball Ball');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 0.1 0.5e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_ball_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_ball_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 0.1 0.5e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_ball_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_ball_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% BALL SPHERE FITTING

% Storage
nb_of_params = 5;
param_07_healthy_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Ball Sphere');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 1e-03 1e-03 0.008 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.5e-03) normrnd(0, 0.007) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Sphere_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.5e-03) normrnd(0, 0.007) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Sphere_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_ball_sphere(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_ball_sphere(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 1e-03 1e-03 0.008 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.5e-03) normrnd(0, 0.007) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Sphere_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 0.5e-03) normrnd(0, 0.5e-03) normrnd(0, 0.007) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Sphere_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_ball_sphere(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_ball_sphere(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% BALL BALL SPHERE FITTING

% Storage
nb_of_params = 8;
param_07_healthy_ball_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_ball_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_ball_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_ball_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',5000,'Algorithm','interior-point','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Ball Ball Sphere');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 0.1e-03 0.1e-03 0.009 0.33 0.33 0.33];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03) normrnd(0, 0.1e-03) normrnd(0, 0.007) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03) normrnd(0, 0.1e-03) normrnd(0, 0.007) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_ball_ball_sphere(ii, jj, :) = best_parameter_hat;
            % nb_of_params-1 bc we can deduce the last f from the others
            % (f=1-f1-f2)
            BIC_07_healthy_ball_ball_sphere(ii, jj) = (nb_of_params-1) * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 0.1e-03 0.1e-03 0.009 0.33 0.33 0.33];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03) normrnd(0, 0.1e-03) normrnd(0, 0.007) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03) normrnd(0, 0.1e-03) normrnd(0, 0.007) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_ball_ball_sphere(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_ball_ball_sphere(ii, jj) = (nb_of_params-1) * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% STICK BALL FITTING

% Storage
nb_of_params = 6;
param_07_healthy_stick_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_stick_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_stick_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_stick_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Stick Ball');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_stick_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_stick_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_stick_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_stick_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% ZEPPELIN BALL FITTING

% Storage
nb_of_params = 7;
param_07_healthy_zeppelin_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_zeppelin_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_zeppelin_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_zeppelin_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Zeppelin Ball');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 50e-03 0 0 0.1e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Zeppelin_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Zeppelin_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_zeppelin_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_zeppelin_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 50e-03 0 0 0.1e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Zeppelin_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Zeppelin_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_zeppelin_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_zeppelin_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% TENSOR BALL FITTING

% Storage
nb_of_params = 8;
param_07_healthy_tensor_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_tensor_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_tensor_ball = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_tensor_ball = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Tensor Ball');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 50e-03 50e-03 0 0 0.1e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03)  normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Tensor_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03)  normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Tensor_Ball_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_tensor_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_tensor_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 50e-03 50e-03 0 0 0.1e-03 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03)  normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Tensor_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03)  normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Tensor_Ball_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_tensor_ball(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_tensor_ball(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% STICK BALL SPHERE FITTING

% Storage
nb_of_params = 10;
param_07_healthy_stick_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_stick_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_stick_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_stick_ball_sphere = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',5000,'Algorithm','interior-point','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Stick Ball Sphere');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 0 0 0.1e-03 0.009 50e-03 0.33 0.33 0.33];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03) normrnd(0, 0.005) normrnd(0, 50e-03) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03) normrnd(0, 0.005) normrnd(0, 50e-03) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_stick_ball_sphere(ii, jj, :) = best_parameter_hat;
            % nb_of_params-1 bc we can deduce the last f from the others
            % (f=1-f1-f2)
            BIC_07_healthy_stick_ball_sphere(ii, jj) = (nb_of_params-1) * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 0 0 0.1e-03 0.009 50e-03 0.33 0.33 0.33];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03) normrnd(0, 0.005) normrnd(0, 50e-03) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03) normrnd(0, 0.005) normrnd(0, 50e-03) normrnd(0, 0.1) normrnd(0, 0.1) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Ball_Sphere_constrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_stick_ball_sphere(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_stick_ball_sphere(ii, jj) = (nb_of_params-1) * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% BALL ZEPPELIN FITTING

% Storage
nb_of_params = 7;
param_07_healthy_ball_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_ball_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_ball_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_ball_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Ball Zeppelin');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 0.1e-03 0.1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_ball_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_ball_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 0.1e-03 0.1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_ball_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_ball_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% BALL TENSOR FITTING

% Storage
nb_of_params = 8;
param_07_healthy_ball_tensor = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_ball_tensor = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_ball_tensor = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_ball_tensor = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Ball Tensor');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 0.1e-03 50e-03 50e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Tensor_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Tensor_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_ball_tensor(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_ball_tensor(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 0.1e-03 50e-03 50e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Ball_Tensor_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:3

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 0.1e-03)  normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Ball_Tensor_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_ball_tensor(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_ball_tensor(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% STICK ZEPPELIN FITTING

% Storage
nb_of_params = 9;
param_07_healthy_stick_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_stick_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_stick_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_stick_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Stick Zeppelin');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 0 0 0.5e-03 0.5e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.5e-03)  normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.5e-03)  normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_stick_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_stick_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 0 0 0.5e-03 0.5e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.5e-03)  normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Stick_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.5e-03)  normrnd(0, 0.5e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Stick_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_stick_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_stick_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;


%% ZEPPELIN ZEPPELIN FITTING

% Storage
nb_of_params = 10;
param_07_healthy_zeppelin_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_healthy_zeppelin_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));
param_07_preecl_zeppelin_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2), nb_of_params);
BIC_07_preecl_zeppelin_zeppelin = zeros(size(healthy_07, 1), size(healthy_07, 2));

% fminunc parameters
h=optimset('MaxFunEvals',8000,'Algorithm','quasi-newton','TolX',1e-6,'TolFun',1e-6,'display','off');
disp('Zeppelin Zeppelin');
tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(healthy_07(ii, jj, :));

            % Start-point estimate
            startx = [0.1 50e-03 50e-03 0 0 0.1e-03 0.1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Zeppelin_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                    ouaisok = false;
                catch err
                    disp('Erreur premier');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2
                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Zeppelin_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(healthy_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième');
                    end
                end
                     
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_healthy_zeppelin_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_healthy_zeppelin_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)

            % We are working on this one
            voxOfInterest = squeeze(preecl_07(ii, jj, :));

            % Start-point estimate
            startx = [0.06 50e-03 50e-03 0 0 0.1e-03 0.1e-03 0 0 0.5];

            % Optimization
            ouaisok = true;
            while(ouaisok)
                try
                    startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                    [parameter_hat,RESNORM,~,~]=Zeppelin_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);                    
                    ouaisok = false;
                catch err
                    disp('Erreur premier preecl');
                end
            end

            best_parameter_hat = parameter_hat;
            best_RESNORM = RESNORM;

            for kk=1:2

                ouaisok = true;
                while(ouaisok)
                    try
                        startx = startx + [normrnd(0, 0.1) normrnd(0, 50e-03) normrnd(0, 50e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1e-03)  normrnd(0, 0.1e-03) normrnd(0, 10) normrnd(0, 10) normrnd(0, 0.1)];
                        [parameter_hat,RESNORM,~,~]=Zeppelin_Zeppelin_unconstrained(startx, voxOfInterest, squeeze(preecl_07(ii, jj, 1)), gradecho, h);
                        ouaisok = false;
                    catch err
                        disp('Erreur deuxième preecl');
                    end
                end
                
                if(RESNORM<best_RESNORM)
                    best_parameter_hat = parameter_hat;
                    best_RESNORM = RESNORM;
                end

            end

            % Storing
            param_07_preecl_zeppelin_zeppelin(ii, jj, :) = best_parameter_hat;
            BIC_07_preecl_zeppelin_zeppelin(ii, jj) = nb_of_params * log(330) + 330 * log((1/330)*best_RESNORM);

        end
    end
end
toc;

%% Verification of the maps

figure();
subplot(121);
imagesc(BIC_07_healthy_zeppelin_zeppelin);
title('Healthy');
subplot(122);
imagesc(BIC_07_preecl_zeppelin_zeppelin);
title('Pre-eclampsia');


%% BIC maps

BIC_map_healthy_07 = zeros(size(healthy_07, 1), size(healthy_07, 2), 3);
BIC_map_preecl_07 = zeros(size(preecl_07, 1), size(preecl_07, 2), 3);
BIC_nb_map_healthy_07 = zeros(1, 11);
BIC_nb_map_preecl_07 = zeros(1, 11);

% 1: ball_ball, RED, iso-iso
% 2: ball_ball_sphere, BLUE, iso-iso
% 3: ball_sphere, GREEN, iso-iso
% 4: ball_tensor, CYAN, iso-aniso
% 5: ball_zeppelin, YELLOW, iso-aniso
% 6: stick_ball, ORANGE, aniso-iso
% 7: stick_ball_sphere, PINK, aniso-iso
% 8: stick_zeppelin, PURPLE, aniso-aniso
% 9: tensor_ball, DARK GREEN, aniso-iso
% 10: zeppelin_ball, MAROON, aniso-iso
% 11: zeppelin_zeppelin, WHITE, aniso-aniso

BIC_colors = [255 0 0; 0 0 255; 0 255 0; 0 213 255; 196 255 0; 229 176 0; 255 117 247; 86 13 255; 0 149 25; 144 115 0; 255 255 255]./255;

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)

        % Verifying if we're in the ROI
        if(mask_healthy_07(ii, jj) ~= 0)
            %BIC_array = [BIC_07_healthy_ball_ball(ii, jj) BIC_07_healthy_ball_ball_sphere(ii, jj) BIC_07_healthy_ball_sphere(ii, jj) BIC_07_healthy_ball_tensor(ii, jj) BIC_07_healthy_ball_zeppelin(ii, jj) BIC_07_healthy_stick_ball(ii, jj) BIC_07_healthy_stick_ball_sphere(ii, jj) BIC_07_healthy_stick_zeppelin(ii, jj) BIC_07_healthy_tensor_ball(ii, jj) BIC_07_healthy_zeppelin_ball(ii, jj) BIC_07_healthy_zeppelin_zeppelin(ii, jj)];
            BIC_array = [Inf Inf Inf BIC_07_healthy_ball_tensor(ii, jj) BIC_07_healthy_ball_zeppelin(ii, jj) BIC_07_healthy_stick_ball(ii, jj) BIC_07_healthy_stick_ball_sphere(ii, jj) BIC_07_healthy_stick_zeppelin(ii, jj) BIC_07_healthy_tensor_ball(ii, jj) BIC_07_healthy_zeppelin_ball(ii, jj) BIC_07_healthy_zeppelin_zeppelin(ii, jj)];
            [~, BIC_min_idx] = min(BIC_array);
            BIC_map_healthy_07(ii, jj, :) = BIC_colors(BIC_min_idx, :);
            BIC_nb_map_healthy_07(1, BIC_min_idx) = BIC_nb_map_healthy_07(1, BIC_min_idx) + 1;
        else
            BIC_map_healthy_07(ii, jj, :) = [healthy_07(ii, jj, 1) healthy_07(ii, jj, 1) healthy_07(ii, jj, 1)]./255;
        end
    end
end
toc;
tic;
for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)

        % Verifying if we're in the ROI
        if(mask_preecl_07(ii, jj) ~= 0)
            %BIC_array = [BIC_07_preecl_ball_ball(ii, jj) BIC_07_preecl_ball_ball_sphere(ii, jj) BIC_07_preecl_ball_sphere(ii, jj) BIC_07_preecl_ball_tensor(ii, jj) BIC_07_preecl_ball_zeppelin(ii, jj) BIC_07_preecl_stick_ball(ii, jj) BIC_07_preecl_stick_ball_sphere(ii, jj) BIC_07_preecl_stick_zeppelin(ii, jj) BIC_07_preecl_tensor_ball(ii, jj) BIC_07_preecl_zeppelin_ball(ii, jj) BIC_07_preecl_zeppelin_zeppelin(ii, jj)];
            BIC_array = [Inf Inf Inf BIC_07_preecl_ball_tensor(ii, jj) BIC_07_preecl_ball_zeppelin(ii, jj) BIC_07_preecl_stick_ball(ii, jj) BIC_07_preecl_stick_ball_sphere(ii, jj) BIC_07_preecl_stick_zeppelin(ii, jj) BIC_07_preecl_tensor_ball(ii, jj) BIC_07_preecl_zeppelin_ball(ii, jj) BIC_07_preecl_zeppelin_zeppelin(ii, jj)];
            [~, BIC_min_idx] = min(BIC_array);
            BIC_map_preecl_07(ii, jj, :) = BIC_colors(BIC_min_idx, :);
            BIC_nb_map_preecl_07(1, BIC_min_idx) = BIC_nb_map_preecl_07(1, BIC_min_idx) + 1;
        else
            BIC_map_preecl_07(ii, jj, :) = [preecl_07(ii, jj, 1) preecl_07(ii, jj, 1) preecl_07(ii, jj, 1)]./255;
        end
    end
end
toc;


%% Visualizing

% BIC map
figure();
subplot(121);
E = double(bwperim(mask_placenta_healthy_07));
imshow(imoverlay(BIC_map_healthy_07, E, 'black'));
hold on;
for ll=1:11
    scatter(0, 0, [], BIC_colors(ll, :), 'filled');
    hold on;
end
scatter(0, 0, [], "black");
hold off;
legend('Ball Ball', 'Ball Ball Sphere', 'Ball Sphere', 'Ball Tensor', 'Ball Zeppelin', 'Stick Ball', 'Stick Ball Sphere', 'Stick Zeppelin', 'Tensor Ball', 'Zeppelin Ball', 'Zeppelin Zeppelin', 'Limit placenta');
title('Healthy control')
subplot(122);
E = double(bwperim(mask_placenta_preecl_07));
imshow(imoverlay(BIC_map_preecl_07, E, 'black'));
hold on;
for ll=1:11
    scatter(0, 0, [], BIC_colors(ll, :), 'filled');
    hold on;
end
scatter(0, 0, [], "black");
hold off;
legend('Ball Ball', 'Ball Ball Sphere', 'Ball Sphere', 'Ball Tensor', 'Ball Zeppelin', 'Stick Ball', 'Stick Ball Sphere', 'Stick Zeppelin', 'Tensor Ball', 'Zeppelin Ball', 'Zeppelin Zeppelin', 'Limit placenta');
title('Pre-eclampsia patient')
sgtitle('BIC maps for the slice 07');


%% Visualizing

% Bar plot
figure();
X_BIC = categorical({'Healthy control','Pre-eclampsia patient'});
X_BIC = reordercats(X_BIC,{'Healthy control','Pre-eclampsia patient'});
bar(X_BIC, [BIC_nb_map_healthy_07; BIC_nb_map_preecl_07]);
ylabel('Number of voxels');
legend('Ball Ball', 'Ball Ball Sphere', 'Ball Sphere', 'Ball Tensor', 'Ball Zeppelin', 'Stick Ball', 'Stick Ball Sphere', 'Stick Zeppelin', 'Tensor Ball', 'Zeppelin Ball', 'Zeppelin Zeppelin');


%% Visualizing

% T2 map
T2_all_map_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2), 11);
T2_all_map_healthy(:, :, 1) = param_07_healthy_ball_ball(:, :, 1);
T2_all_map_healthy(:, :, 2) = param_07_healthy_ball_ball_sphere(:, :, 1);
T2_all_map_healthy(:, :, 3) = param_07_healthy_ball_sphere(:, :, 1);
T2_all_map_healthy(:, :, 4) = param_07_healthy_ball_tensor(:, :, 1);
T2_all_map_healthy(:, :, 5) = param_07_healthy_ball_zeppelin(:, :, 1);
T2_all_map_healthy(:, :, 6) = param_07_healthy_stick_ball(:, :, 1);
T2_all_map_healthy(:, :, 7) = param_07_healthy_stick_ball_sphere(:, :, 1);
T2_all_map_healthy(:, :, 8) = param_07_healthy_stick_zeppelin(:, :, 1);
T2_all_map_healthy(:, :, 9) = param_07_healthy_tensor_ball(:, :, 1);
T2_all_map_healthy(:, :, 10) = param_07_healthy_zeppelin_ball(:, :, 1);
T2_all_map_healthy(:, :, 11) = param_07_healthy_zeppelin_zeppelin(:, :, 1);

T2_all_map_preecl = zeros(size(healthy_07, 1), size(healthy_07, 2), 11);
T2_all_map_preecl(:, :, 1) = param_07_preecl_ball_ball(:, :, 1);
T2_all_map_preecl(:, :, 2) = param_07_preecl_ball_ball_sphere(:, :, 1);
T2_all_map_preecl(:, :, 3) = param_07_preecl_ball_sphere(:, :, 1);
T2_all_map_preecl(:, :, 4) = param_07_preecl_ball_tensor(:, :, 1);
T2_all_map_preecl(:, :, 5) = param_07_preecl_ball_zeppelin(:, :, 1);
T2_all_map_preecl(:, :, 6) = param_07_preecl_stick_ball(:, :, 1);
T2_all_map_preecl(:, :, 7) = param_07_preecl_stick_ball_sphere(:, :, 1);
T2_all_map_preecl(:, :, 8) = param_07_preecl_stick_zeppelin(:, :, 1);
T2_all_map_preecl(:, :, 9) = param_07_preecl_tensor_ball(:, :, 1);
T2_all_map_preecl(:, :, 10) = param_07_preecl_zeppelin_ball(:, :, 1);
T2_all_map_preecl(:, :, 11) = param_07_preecl_zeppelin_zeppelin(:, :, 1);


T2_best_BIC_healthy = zeros(size(healthy_07, 1), size(healthy_07, 2));
T2_best_BIC_preecl = zeros(size(preecl_07, 1), size(preecl_07, 2));

tic;
for ii=1:size(healthy_07, 1)
    for jj=1:size(healthy_07, 2)
        if(mask_healthy_07(ii, jj) ~= 0)
            %BIC_array = [BIC_07_preecl_ball_ball(ii, jj) BIC_07_preecl_ball_ball_sphere(ii, jj) BIC_07_preecl_ball_sphere(ii, jj) BIC_07_preecl_ball_tensor(ii, jj) BIC_07_preecl_ball_zeppelin(ii, jj) BIC_07_preecl_stick_ball(ii, jj) BIC_07_preecl_stick_ball_sphere(ii, jj) BIC_07_preecl_stick_zeppelin(ii, jj) BIC_07_preecl_tensor_ball(ii, jj) BIC_07_preecl_zeppelin_ball(ii, jj) BIC_07_preecl_zeppelin_zeppelin(ii, jj)];
            BIC_array = [Inf Inf Inf BIC_07_healthy_ball_tensor(ii, jj) BIC_07_healthy_ball_zeppelin(ii, jj) BIC_07_healthy_stick_ball(ii, jj) BIC_07_healthy_stick_ball_sphere(ii, jj) BIC_07_healthy_stick_zeppelin(ii, jj) BIC_07_healthy_tensor_ball(ii, jj) BIC_07_healthy_zeppelin_ball(ii, jj) BIC_07_healthy_zeppelin_zeppelin(ii, jj)];
            [~, BIC_min_idx] = min(BIC_array);
            T2_best_BIC_healthy(ii, jj) = exp(-(T2_all_map_healthy(ii, jj, BIC_min_idx).^2))*0.25 ;
        end
    end
end

for ii=1:size(preecl_07, 1)
    for jj=1:size(preecl_07, 2)
        if(mask_preecl_07(ii, jj) ~= 0)
            %BIC_array = [BIC_07_preecl_ball_ball(ii, jj) BIC_07_preecl_ball_ball_sphere(ii, jj) BIC_07_preecl_ball_sphere(ii, jj) BIC_07_preecl_ball_tensor(ii, jj) BIC_07_preecl_ball_zeppelin(ii, jj) BIC_07_preecl_stick_ball(ii, jj) BIC_07_preecl_stick_ball_sphere(ii, jj) BIC_07_preecl_stick_zeppelin(ii, jj) BIC_07_preecl_tensor_ball(ii, jj) BIC_07_preecl_zeppelin_ball(ii, jj) BIC_07_preecl_zeppelin_zeppelin(ii, jj)];
            BIC_array = [Inf Inf Inf BIC_07_preecl_ball_tensor(ii, jj) BIC_07_preecl_ball_zeppelin(ii, jj) BIC_07_preecl_stick_ball(ii, jj) BIC_07_preecl_stick_ball_sphere(ii, jj) BIC_07_preecl_stick_zeppelin(ii, jj) BIC_07_preecl_tensor_ball(ii, jj) BIC_07_preecl_zeppelin_ball(ii, jj) BIC_07_preecl_zeppelin_zeppelin(ii, jj)];
            [~, BIC_min_idx] = min(BIC_array);
            T2_best_BIC_preecl(ii, jj) = exp(-(T2_all_map_preecl(ii, jj, BIC_min_idx).^2))*0.25 ;
        end
    end
end
toc;


%% Visualizing (T2)

cmap = colormap('hot');

figure();
subplot(141);
to_plot = T2_best_BIC_healthy;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('Healthy T_2^{*} map');
subplot(142);
boxplot(reshape(T2_best_BIC_healthy, [], 1));
ylim([0.02 0.28]);
ylabel('T_2^* (s)')
title('Healthy T_2^{*} boxplot');
subplot(143);
to_plot = T2_best_BIC_preecl;
to_plot = to_plot/max(to_plot(:));
imshow(to_plot, Colormap=cmap);
title('Pre-eclampsia T_2^{*} map');
subplot(144);
boxplot(reshape(T2_best_BIC_preecl, [], 1));
ylim([0.02 0.28]);
ylabel('T_2^* (s)')
title('Pre-eclampsia T_2^{*} boxplot');


%% Appendix

% fmincon fitting for T2*-IVIM
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = IVIM_constrained(x, Avox, S0, TE, bgrad, Dmax, Dx, h)
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 Dmax/Dx 0];
    ub = [0.25 Dmax/Dx Dmax 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        D = x(2);
        Dp = x(3);
        f = x(4);
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( f*exp(-bgrad*Dp) + (1-f)*exp(-bgrad*D) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc fitting for T2*-IVIM
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = IVIM_unconstrained(x, Avox, S0, TE, bgrad, Dmax, Dx, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        D = exp(-(x(2).^2))*(Dmax/Dx);
        Dp = exp(-(x(3).^2))*(Dmax - (Dmax/Dx)) + (Dmax/Dx);
        f = exp(-(x(4).^2));
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( f*exp(-bgrad*Dp) + (1-f)*exp(-bgrad*D) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for the ADC model fitting
function SSD = ADC_SSD(Avox, S0, TE, T2_star, bgrad, D)
    S = S0 * exp(-TE/T2_star) .* exp(-bgrad*D);
    SSD = sum((Avox - S).^2);
end

% Error function for the IVIM model fitting
function SSD = IVIM_SSD(Avox, S0, TE, T2_star, bgrad, D, Dp, f)
    S = S0 * exp(-TE/T2_star) .* ( f * exp(-bgrad*Dp) + (1-f) * exp(-bgrad*D) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                             BALL BALL                                %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Ball-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Ball_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 0];
    ub = [0.25 1000e-03 5e-03 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIMM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIMM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        fv = x(4);
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.')) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Ball-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Ball_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(4).^2));
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.')) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Ball-Ball
function SSD = Ball_Ball_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(4).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        fv = x(4);
    end
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.')) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                             BALL SPHERE                              %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Ball-Sphere
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Sphere_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0.01e-03 0.01e-03 0.001 0];
    ub = [0.25 1000e-03 1000e-03 0.02 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        Dsphere = x(3);
        r = x(4);
        fsphere = x(5);
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( (1-fsphere)*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + fsphere*r*exp(-bvals.*Dsphere) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Ball-Sphere
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Sphere_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 0.01e-03 + exp(-(x(2).^2))*(1000e-03 - 0.01e-03);
        Dsphere = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        r = 0.001 + exp(-(x(4).^2))*(0.02 - 0.001);
        fsphere = exp(-(x(5).^2));
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( (1-fsphere)*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + fsphere*r*exp(-bvals.*Dsphere) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Ball-Sphere
function SSD = Ball_Sphere_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 0.01e-03 + exp(-(x(2).^2))*(1000e-03 - 0.01e-03);
        Dsphere = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        r = 0.001 + exp(-(x(4).^2))*(0.02 - 0.001);
        fsphere = exp(-(x(5).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        Dsphere = x(3);
        r = x(4);
        fsphere = x(5);
    end
        S = S0 * exp(-TE/T2_star) .* ( (1-fsphere)*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + fsphere*r*exp(-bvals.*Dsphere) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                          BALL BALL SPHERE                            %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Ball-Ball-Sphere
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Ball_Sphere_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [0, 0, 0, 0, 0, 1, 1, 1];
    beq = 1;
    lb = [0 5e-03 0.01e-03 0.01e-03 0.001 0 0 0];
    ub = [0.25 1000e-03 5e-03 5e-03 0.02 1 1 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        Dsphere = x(4);
        r = x(5);
        fv = x(6);
        fsphere = x(7);
        f = x(8);
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + f*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) + fsphere*r*exp(-bvals*Dsphere) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end


% Error function for Ball-Ball-Sphere
function SSD = Ball_Ball_Sphere_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        disp('Impossible here');
    else
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        Dsphere = x(4);
        r = x(5);
        fv = x(6);
        fsphere = x(7);
        f = x(8);
    end
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + f*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) + fsphere*r*exp(-bvals*Dsphere) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                             STICK BALL                               %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Stick-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Stick_Ball_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 -100 -100 0];
    ub = [0.25 1000e-03 5e-03 100 100 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        phi = x(4);
        theta = x(5);
        fv = x(6);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* (grad*n.').^2 ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Stick-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Stick_Ball_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(4).^2))*(100 + 100);
        theta = -100 + exp(-(x(5).^2))*(100 + 100);
        fv = exp(-(x(6).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* (grad*n.').^2 ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Stick-Ball
function SSD = Stick_Ball_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D = 0.01e-03 + exp(-(x(3).^2))*(5-03 - 0.01e-03);
        phi = -100 + exp(-(x(4).^2))*(100 + 100);
        theta = -100 + exp(-(x(5).^2))*(100 + 100);
        fv = exp(-(x(6).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        D = x(3);
        phi = x(4);
        theta = x(5);
        fv = x(6);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* (grad*n.').^2 ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                           ZEPPELIN BALL                              %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Zeppelin-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Zeppelin_Ball_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 -100 -100 0.01e-03 0];
    ub = [0.25 1000e-03 1000e-03 100 100 5e-03 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv_para = x(2);
        Dv_perp = x(3);
        phi = x(4);
        theta = x(5);
        D = x(6);
        fv = x(7);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals.* diag(grad* ( (Dv_para - Dv_perp) * (n.' * n) + Dv_perp * eye(3) ) *grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Zeppelin-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Zeppelin_Ball_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(4).^2))*(100 + 100);
        theta = -100 + exp(-(x(5).^2))*(100 + 100);
        D = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(7).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals.* diag(grad* ( (Dv_para - Dv_perp) * (n.' * n) + Dv_perp * eye(3) ) *grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Zeppelin_Ball
function SSD = Zeppelin_Ball_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(4).^2))*(100 + 100);
        theta = -100 + exp(-(x(5).^2))*(100 + 100);
        D = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(7).^2));
    else
        T2_star = x(1);
        Dv_para = x(2);
        Dv_perp = x(3);
        phi = x(4);
        theta = x(5);
        D = x(6);
        fv = x(7);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals.* diag(grad* ( (Dv_para - Dv_perp) * (n.' * n) + Dv_perp * eye(3) ) *grad.')) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                           TENSOR BALL                                %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Tensor-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Tensor_Ball_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 0.01e-03 -100 -100 0.01e-03 0];
    ub = [0.25 1000e-03 1000e-03 1000e-03 100 100 5e-03 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv_para = x(2);
        Dv1_perp = x(3);
        Dv2_perp = x(4);
        phi = x(5);
        theta = x(6);
        D = x(7);
        fv = x(8);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
        n_perp2 = [-sin(phi) cos(phi) 0];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( Dv_para*(n.'*n) + Dv1_perp*(n_perp1.'*n_perp1) + Dv2_perp*(n_perp2.'*n_perp2) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Tensor-Ball
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Tensor_Ball_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv1_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        Dv2_perp = 0.01e-03 + exp(-(x(4).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(5).^2))*(100 + 100);
        theta = -100 + exp(-(x(6).^2))*(100 + 100);
        D = 0.01e-03 + exp(-(x(7).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(8).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
        n_perp2 = [-sin(phi) cos(phi) 0];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( Dv_para*(n.'*n) + Dv1_perp*(n_perp1.'*n_perp1) + Dv2_perp*(n_perp2.'*n_perp2) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Tensor-Ball
function SSD = Tensor_Ball_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv1_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        Dv2_perp = 0.01e-03 + exp(-(x(4).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(5).^2))*(100 + 100);
        theta = -100 + exp(-(x(6).^2))*(100 + 100);
        D = 0.01e-03 + exp(-(x(7).^2))*(5e-03 - 0.01e-03);
        fv = exp(-(x(8).^2));
    else
        T2_star = x(1);
        Dv_para = x(2);
        Dv1_perp = x(3);
        Dv2_perp = x(4);
        phi = x(5);
        theta = x(6);
        D = x(7);
        fv = x(8);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
    n_perp2 = [-sin(phi) cos(phi) 0];
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( Dv_para*(n.'*n) + Dv1_perp*(n_perp1.'*n_perp1) + Dv2_perp*(n_perp2.'*n_perp2) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                         STICK BALL SPHERE                            %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Stick-Ball-Sphere
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Stick_Ball_Sphere_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    beq = 1;
    lb = [0 5e-03 -100 -100 0.01e-03 0.001 0.01e-03 0 0 0];
    ub = [0.25 1000e-03 100 100 5e-03 0.02 1000e-03 1 1 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        phi = x(3);
        theta = x(4);
        D = x(5);
        r = x(6);
        Dsphere = x(7);
        fv = x(8);
        fsphere = x(9);
        f = x(10);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* ((grad*n.').^2) ) + f*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) + fsphere*r*exp(-bvals.*Dsphere) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end


% Error function for Stick-Ball-Sphere
function SSD = Stick_Ball_Sphere_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        disp('Impossible here');
    else
        T2_star = x(1);
        Dv = x(2);
        phi = x(3);
        theta = x(4);
        D = x(5);
        r = x(6);
        Dsphere = x(7);
        fv = x(8);
        fsphere = x(9);
        f = x(10);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    % Synthesize the signals according to the model
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* ((grad*n.').^2) ) + f*exp(-bvals .* diag(grad*(D*eye(3))*grad.') ) + fsphere*r*exp(-bvals.*Dsphere) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                           BALL ZEPPELIN                              %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Ball-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Zeppelin_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 0.01e-03 -100 -100 0];
    ub = [0.25 1000e-03 5e-03 5e-03 100 100 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        D_para = x(3);
        D_perp = x(4);
        phi = x(5);
        theta = x(6);
        fv = x(7);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Ball-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Zeppelin_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D_para = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(4).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(5).^2))*(100 + 100);
        theta = -100 + exp(-(x(6).^2))*(100 + 100);
        fv = exp(-(x(7).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Ball-Zeppelin
function SSD = Ball_Zeppelin_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D_para = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(4).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(5).^2))*(100 + 100);
        theta = -100 + exp(-(x(6).^2))*(100 + 100);
        fv = exp(-(x(7).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        D_para = x(3);
        D_perp = x(4);
        phi = x(5);
        theta = x(6);
        fv = x(7);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            BALL TENSOR                               %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Ball-Tensor
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Tensor_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 0.01e-03 0.01e-03 -100 -100 0];
    ub = [0.25 1000e-03 5e-03 1000e-03 1000e-03 100 100 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        D_para = x(3);
        D_perp1 = x(4);
        D_perp2 = x(5);
        phi = x(6);
        theta = x(7);
        fv = x(8);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
        n_perp2 = [-sin(phi) cos(phi) 0];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( D_para*(n.'*n) + D_perp1*(n_perp1.'*n_perp1) + D_perp2*(n_perp2.'*n_perp2) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Ball-Tensor
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Ball_Tensor_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D_para = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        D_perp1 = 0.01e-03 + exp(-(x(4).^2))*(1000e-03 - 0.01e-03);
        D_perp2 = 0.01e-03 + exp(-(x(5).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(6).^2))*(100 + 100);
        theta = -100 + exp(-(x(7).^2))*(100 + 100);
        fv = exp(-(x(8).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
        n_perp2 = [-sin(phi) cos(phi) 0];
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( D_para*(n.'*n) + D_perp1*(n_perp1.'*n_perp1) + D_perp2*(n_perp2.'*n_perp2) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Ball-Tensor
function SSD = Ball_Tensor_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        D_para = 0.01e-03 + exp(-(x(3).^2))*(5e-03 - 0.01e-03);
        D_perp1 = 0.01e-03 + exp(-(x(4).^2))*(1000e-03 - 0.01e-03);
        D_perp2 = 0.01e-03 + exp(-(x(5).^2))*(1000e-03 - 0.01e-03);
        phi = -100 + exp(-(x(6).^2))*(100 + 100);
        theta = -100 + exp(-(x(7).^2))*(100 + 100);
        fv = exp(-(x(8).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        D_para = x(3);
        D_perp1 = x(4);
        D_perp2 = x(5);
        phi = x(6);
        theta = x(7);
        fv = x(8);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    n_perp1 = [cos(phi)*cos(theta) sin(phi)*cos(theta) -sin(theta)];
    n_perp2 = [-sin(phi) cos(phi) 0];
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad*(Dv*eye(3))*grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( D_para*(n.'*n) + D_perp1*(n_perp1.'*n_perp1) + D_perp2*(n_perp2.'*n_perp2) ) *grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                           STICK ZEPPELIN                             %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Stick-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Stick_Zeppelin_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 -100 -100 0.01e-03 0.01e-03 -100 -100 0];
    ub = [0.25 1000e-03 100 100 5e-03 5e-03 100 100 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv = x(2);
        phiv = x(3);
        thetav = x(4);
        D_para = x(5);
        D_perp = x(6);
        phi = x(7);
        theta = x(8);
        fv = x(9);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* ((grad*nv.').^2) )+ (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Stick-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Stick_Zeppelin_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        phiv = -100 + exp(-(x(3).^2))*(100 + 100);
        thetav = -100 + exp(-(x(4).^2))*(100 + 100);
        D_para = 0.01e-03 + exp(-(x(5).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(7).^2))*(100 + 100);
        theta = -100 + exp(-(x(8).^2))*(100 + 100);
        fv = exp(-(x(9).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* ((grad*nv.').^2) )+ (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Stick-Zeppelin
function SSD = Stick_Zeppelin_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        phiv = -100 + exp(-(x(3).^2))*(100 + 100);
        thetav = -100 + exp(-(x(4).^2))*(100 + 100);
        D_para = 0.01e-03 + exp(-(x(5).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(7).^2))*(100 + 100);
        theta = -100 + exp(-(x(8).^2))*(100 + 100);
        fv = exp(-(x(9).^2));
    else
        T2_star = x(1);
        Dv = x(2);
        phiv = x(3);
        thetav = x(4);
        D_para = x(5);
        D_perp = x(6);
        phi = x(7);
        theta = x(8);
        fv = x(9);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-Dv* bvals .* ((grad*nv.').^2) )+ (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
    SSD = sum((Avox - S).^2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                        ZEPPELIN ZEPPELIN                             %%            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fmincon for Zeppelin-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Zeppelin_Zeppelin_constrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Parameters for fmincon
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 5e-03 0.01e-03 -100 -100 0.01e-03 0.01e-03 -100 -100 0];
    ub = [0.25 1000e-03 1000e-03 100 100 5e-03 5e-03 100 100 1];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@IVIM, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = x(1);
        Dv_para = x(2);
        Dv_perp = x(3);
        phiv = x(4);
        thetav = x(5);
        D_para = x(6);
        D_perp = x(7);
        phi = x(8);
        theta = x(9);
        fv = x(10);
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( (Dv_para - Dv_perp) * (nv.' * nv) + Dv_perp * eye(3) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% fminunc for Zeppelin-Zeppelin
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = Zeppelin_Zeppelin_unconstrained(x, Avox, S0, gradecho, h)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Extract the parameters
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        phiv = -100 + exp(-(x(4).^2))*(100 + 100);
        thetav = -100 + exp(-(x(5).^2))*(100 + 100);
        D_para = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(7).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(8).^2))*(100 + 100);
        theta = -100 + exp(-(x(9).^2))*(100 + 100);
        fv = exp(-(x(10).^2));
        % Fibre direction
        n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
        % Synthesize the signals according to the model
        S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( (Dv_para - Dv_perp) * (nv.' * nv) + Dv_perp * eye(3) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
        % Compute the sum of square differences
        sumRes = sum((Avox - S).^2);
    end
end

% Error function for Zeppelin-Zeppelin
function SSD = Zeppelin_Zeppelin_SSD(Avox, S0, gradecho, x, unc_yes)
    % Extracting information from gradecho
    grad = gradecho(:, 1:3);
    bvals = gradecho(:, 4);
    TE = gradecho(:, 5);
    % Formatting x
    if(unc_yes)
        T2_star = exp(-(x(1).^2))*0.25;
        Dv_para = 5e-03 + exp(-(x(2).^2))*(1000e-03 - 5e-03);
        Dv_perp = 0.01e-03 + exp(-(x(3).^2))*(1000e-03 - 0.01e-03);
        phiv = -100 + exp(-(x(4).^2))*(100 + 100);
        thetav = -100 + exp(-(x(5).^2))*(100 + 100);
        D_para = 0.01e-03 + exp(-(x(6).^2))*(5e-03 - 0.01e-03);
        D_perp = 0.01e-03 + exp(-(x(7).^2))*(5e-03 - 0.01e-03);
        phi = -100 + exp(-(x(8).^2))*(100 + 100);
        theta = -100 + exp(-(x(9).^2))*(100 + 100);
        fv = exp(-(x(10).^2));
    else
        T2_star = x(1);
        Dv_para = x(2);
        Dv_perp = x(3);
        phiv = x(4);
        thetav = x(5);
        D_para = x(6);
        D_perp = x(7);
        phi = x(8);
        theta = x(9);
        fv = x(10);
    end
    % Fibre direction
    n = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    nv = [cos(phiv)*sin(thetav) sin(phiv)*sin(thetav) cos(thetav)]; 
    S = S0 * exp(-TE/T2_star) .* ( fv*exp(-bvals .* diag(grad* ( (Dv_para - Dv_perp) * (nv.' * nv) + Dv_perp * eye(3) ) *grad.') ) + (1-fv)*exp(-bvals .* diag(grad* ( (D_para - D_perp) * (n.' * n) + D_perp * eye(3) ) *grad.') ) );
    SSD = sum((Avox - S).^2);
end






