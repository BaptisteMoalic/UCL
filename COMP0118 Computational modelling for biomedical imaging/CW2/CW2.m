close all;
clear;
clc;

%%% COMP0118 
%%% Coursework 2 


%% PART 1
%% Sampling
%% 1a - fixing the parameters, generating data

% Parameters of our simulation
sample_size = 20;
mean_1 = 1.5;
mean_2 = 2;
std_sample = 0.2;
rng(22182376);

sample_1 = normrnd(mean_1, std_sample, 20, 1);
sample_2 = normrnd(mean_2, std_sample, 20, 1);


%% 1a - verifying the data

disp('Sample 1');
disp('Mean : '+string(mean(sample_1)));
disp('Std : '+string(std(sample_1)));
disp('');
disp('Sample 2');
disp('Mean : '+string(mean(sample_2)));
disp('Std : '+string(std(sample_2)));


%% 1b - ttest2

disp('The result of ttest2 is: '+string(ttest2(sample_1, sample_2)));


%% 1ci - design matrix

ones_vector = ones(20, 1);
zeros_vector = zeros(20, 1);

X = [ones_vector zeros_vector; zeros_vector ones_vector];


%% 1cii - perpendicular projection operator

PX = (1/20)*[ones(20, 20) zeros(20, 20); zeros(20, 20) ones(20, 20)];


%% 1ciii - Y hat

Y_gt = [sample_1; sample_2];
e = normrnd(0, 1, 40, 1); % noise
Y = Y_gt + e;

Y_hat = PX * Y;


%% 1civ - Rx

RX = (1/20)*[20*eye(20)-ones(20, 20) zeros(20, 20); zeros(20, 20) 20*eye(20)-ones(20, 20)];


%% 1cv - e hat

e_hat = RX * Y;


%% 1cvi - angle Y_hat e_hat

% We start by normalizing the vectors, so we can retrieve the cosine of
% their angle via the dot product

Y_hat_norm = Y_hat/norm(Y_hat);
e_hat_norm = e_hat/norm(e_hat);

cos_angle = Y_hat_norm.' * e_hat_norm;
angle = acos(cos_angle);


%% 1cvii - beta_est

beta_est = (1/20)*X.'*Y;

disp('beta_est = '+string(beta_est));


%% 1cviii - stochastic component

sigma_hat_squared = (e_hat.' * e_hat)/(40-2);

sigma_hat = sqrt(sigma_hat_squared);


%% 1cix - covariance matrix

S_beta = sigma_hat_squared*pinv(X.' * X);


%% 1cxi - reduced model

X_0 = ones(40, 1);
PX_0 = (1/40)*ones(40, 40);
RX_0 = eye(40) - PX_0;

e_hat_0 = RX_0 * Y;

SSR = e_hat.' * e_hat;
SSR_0 = e_hat_0.' * e_hat_0;

% degrees of freedom
nu_1 = trace(PX - PX_0);
nu_2 = trace(eye(40) - PX);

F_stat = ((SSR_0 - SSR)/nu_1) / (SSR/nu_2);


%% 1cxii - empirical t-statistic

lambda_model = [1; -1];

t_stat = (lambda_model.' * beta_est) / (sqrt(lambda_model.' * S_beta * lambda_model));


%% 1cxiv - projection of e in C(X)

proj_e_cx = PX * e;


%% 1cxv - projection of e in C(X)orth

proj_e_cxorth = RX * e;


%% d
%% 1di - X

ones_vector = ones(20, 1);
zeros_vector = zeros(20, 1);

X = [ones_vector ones_vector zeros_vector; ones_vector zeros_vector ones_vector];


%% 1dii - PX

PX = X * pinv(X.' * X) * X.';


%% 1div

RX = eye(40) - PX;

e_hat = RX * Y;

beta_est = pinv(X.' * X)*X.'*Y;

disp('beta_est = '+string(beta_est));

sigma_hat_squared = (e_hat.' * e_hat)/(40-2);

sigma_hat = sqrt(sigma_hat_squared);

S_beta = sigma_hat_squared*pinv(X.' * X);

lambda_model = [0; 1; -1];

t_stat = (lambda_model.' * beta_est) * pinv((sqrt(lambda_model.' * S_beta * lambda_model)));

%% e

ones_vector = ones(20, 1);
zeros_vector = zeros(20, 1);

X = [ones_vector ones_vector zeros_vector; ones_vector zeros_vector ones_vector];

PX = X * pinv(X.' * X) * X.';

RX = eye(40) - PX;

e_hat = RX * Y;

beta_est = pinv(X.' * X)*X.'*Y;

disp('beta_est = '+string(beta_est));

sigma_hat_squared = (e_hat.' * e_hat)/(40-2);

sigma_hat = sqrt(sigma_hat_squared);

S_beta = sigma_hat_squared*pinv(X.' * X);

lambda_model = [0; 0; 1];

t_stat = (lambda_model.' * beta_est) * pinv((sqrt(lambda_model.' * S_beta * lambda_model)));


%% 2ai

disp('The result of ttest is: '+string(ttest(sample_1, sample_2)));


%% 2bi - new GLM model

X = zeros(40, 23);
X(:, 1) = [ones(1, 20) ones(1, 20)].';
X(:, 2) = [ones(1, 20) zeros(1, 20)].';
X(:, 3) = [zeros(1, 20) ones(1, 20)].';
X(1:20, 4:end) = eye(20);
X(21:end, 4:end) = eye(20);


%% 2bii - contrast vector

lambda_model = [0; 1; -1; zeros(20, 1)];


%% 2biii - t-stat

PX = X * pinv(X.' * X) * X.';

RX = eye(40) - PX;

e_hat = RX * Y;

beta_est = pinv(X.' * X)*X.'*Y;

disp('beta_est = '+string(beta_est));

sigma_hat_squared = (e_hat.' * e_hat)/(40-2);

sigma_hat = sqrt(sigma_hat_squared);

S_beta = sigma_hat_squared*pinv(X.' * X);

t_stat = (lambda_model.' * beta_est) * pinv((sqrt(lambda_model.' * S_beta * lambda_model)));




%% PART 2
%% 1a

mean_1 = 1.5;
mean_2 = 2;
std_sample = 0.2;

sample_1 = normrnd(mean_1, std_sample, 6, 1);
sample_2 = normrnd(mean_2, std_sample, 8, 1);

disp('Sample 1');
disp('Mean : '+string(mean(sample_1)));
disp('Std : '+string(std(sample_1)));
disp('');
disp('Sample 2');
disp('Mean : '+string(mean(sample_2)));
disp('Std : '+string(std(sample_2)));


%% 1a - tests

[ttest2_res, ttest2_pval] = ttest2(sample_1, sample_2);

disp('The result of ttest2 is: '+string(ttest2_res));
disp('The p-value is: '+string(ttest2_pval));


%% 1bi

D = [sample_1; sample_2];


%% 1bii - between groups permutations

perms_n1 = nchoosek(D, 6); %permutation for sample size 1 (=6)
perms_n2 = nchoosek(D, 8); %permutation for sample size 2 (=8)

D_perms = zeros(size(perms_n1, 1)*size(perms_n2, 1), 6+8);
idx_D_perms = 1;

for n1=1:size(perms_n1, 1)
    for n2=1:size(perms_n2, 1)
        D_perms(idx_D_perms, :) = [perms_n1(n1, :) perms_n2(n2, :)];
        idx_D_perms = idx_D_perms + 1;
    end
end


%% 1bii - adding noise

D_perms = D_perms + normrnd(0, 1, 9018009, 14);


%% 1biii - computation routine

X = [ones(6, 1) zeros(6, 1); zeros(8, 1) ones(8, 1)];

lambda_model = [1; -1];

PX = X * pinv(X.' * X) * X.';

RX = eye(14) - PX;

beta_vectors = pinv(X.' * X)*X.'*D_perms.';

numerator_full = lambda_model.' * beta_vectors;

res_vector = RX * D_perms';

sigma_squared_not_summed = D_perms.' .* res_vector;

sigma_squared_vector = sum(sigma_squared_not_summed);

pre_denominator = lambda_model.' * pinv(X.' * X) * lambda_model;

denominator_full = sqrt(pre_denominator * sigma_squared_vector);

t_stats_vector = numerator_full ./ denominator_full;


%% 1biii - histogram

hist(t_stats_vector, 50);
title('Empirical distribution of the t-statistic');
xlabel('t-statistic value');
ylabel('Number of occurences');


%% 1biv - p-value

original_t_stat = t_stats_vector(1);

new_p_value = sum((t_stats_vector>=original_t_stat))/(size(perms_n1, 1)*size(perms_n2, 1));


%% 1c - diff of means as a stat

diff_of_means_stat = beta_vectors(1, :) - beta_vectors(2, :);


%% 1c - histogram

hist(diff_of_means_stat, 50);
title('Empirical distribution of the difference of means');
xlabel('Difference of means');
ylabel('Number of occurences');


%% 1c - p-value with diff of means

original_diff_of_means = diff_of_means_stat(1);

new_p_value = sum((diff_of_means_stat>=original_diff_of_means))/(size(perms_n1, 1)*size(perms_n2, 1));


%% 1d - approximate permutation-based p-val

nb_of_perms = 1000;

D_randperm = zeros(nb_of_perms, 14); %storage
D_randperm(1, :) = D.'; %should contain the original non-permuted values

for kk=2:1000
    array_randperm = randperm(14);
    for ll=1:14
        D_randperm(kk, ll) = D(array_randperm(ll));
    end
end

D_randperm = D_randperm + normrnd(0, 1, nb_of_perms, 14);


%% 1d - computation routine

X = [ones(6, 1) zeros(6, 1); zeros(8, 1) ones(8, 1)];
PX = X * pinv(X.' * X) * X.';
RX = eye(14) - PX;
lambda_model = [1; -1];
beta_vectors = pinv(X.' * X)*X.'*D_randperm.';
numerator_full = lambda_model.' * beta_vectors;
res_vector = RX * D_randperm';
sigma_squared_not_summed = D_randperm.' .* res_vector;
sigma_squared_vector = sum(sigma_squared_not_summed);
pre_denominator = lambda_model.' * pinv(X.' * X) * lambda_model;
denominator_full = sqrt(pre_denominator * sigma_squared_vector);
t_stats_vector = numerator_full ./ denominator_full;

%% 1d - new p_value

original_t_stat = t_stats_vector(1);
new_p_value = sum((t_stats_vector>=original_t_stat))/(size(D_randperm, 1));


%% Last question
%% Importing and pre-processing of the data

% Importing the Y data
CPA_idx = [string(0)+string(4); string(0)+string(5); string(0)+string(6); string(0)+string(7); string(0)+string(8); string(0)+string(9); string(10); string(11)];
PPA_idx = [string(0)+string(3); string(0)+string(6); string(0)+string(9); string(10); string(13); string(14); string(15); string(16)];
% We will put the voxel data on top of each other
big_Y = zeros(16, 40, 40, 40);

for kk=1:8
    
    fid = fopen('data/CPA'+CPA_idx(kk)+'_diffeo_fa.img', 'r', 'l'); % little-endian
    dataCPA = fread(fid, 'float'); % 16-bit floating point
    dataCPA = reshape(dataCPA, [40 40 40]); % dimension 40x40x40

    big_Y(kk, :, :, :) = dataCPA;
    
    fid = fopen('data/PPA'+PPA_idx(kk)+'_diffeo_fa.img', 'r', 'l'); % little-endian
    dataPPA = fread(fid, 'float'); % 16-bit floating point
    dataPPA = reshape(dataPPA, [40 40 40]); % dimension 40x40x40

    big_Y(8+kk, :, :, :) = dataPPA;

end

% Importing the mask
fid = fopen('data/wm_mask.img', 'r', 'l'); % little-endian
data_mask = fread(fid, 'float'); % 16-bit floating point
data_mask = reshape(data_mask, [40 40 40]); % dimension 40x40x40


%% Computing the t-statistics

% Invariants of the routine
X = [ones(8, 1) zeros(8, 1); zeros(8, 1) ones(8, 1)];
lambda_model = [1; -1];
PX = X * pinv(X.' * X) * X.';
RX = eye(16) - PX;
pre_denominator = lambda_model.' * pinv(X.' * X) * lambda_model;

% Storage
t_stat_cube = zeros(40, 40, 40);

% Routine
for ii=1:40
    for jj=1:40
        for kk=1:40

            if(data_mask(ii, jj, kk) ~= 0) %ROI
                Y_kk = big_Y(:, ii, jj, kk);
    
                beta_vectors = pinv(X.' * X)*X.'*Y_kk;
                numerator_full = lambda_model.' * beta_vectors;
                res_vector = RX * Y_kk;
                sigma_squared_not_summed = Y_kk .* res_vector;
                sigma_squared_vector = sum(sigma_squared_not_summed);
                denominator_full = sqrt(pre_denominator * sigma_squared_vector);
                t_stats_vector = numerator_full ./ denominator_full;
                t_stat_cube(ii, jj, kk) = t_stats_vector;
            end

        end
    end
end


%% Retrieving the biggest one

max_t_stat_all_voxels = max(abs(t_stat_cube(:)));


%% Permutations 

nb_of_perms = 10000;

D_randperm = zeros(nb_of_perms, 16); %storage
D_randperm(1, :) = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]; %should contain the original non-permuted values

tic;
for kk=2:nb_of_perms
    D_randperm(kk, :) = randperm(16);
end
toc;


%% Routine but with the permutations

% Invariants of the routine
X = [ones(8, 1) zeros(8, 1); zeros(8, 1) ones(8, 1)];
lambda_model = [1; -1];
PX = X * pinv(X.' * X) * X.';
RX = eye(16) - PX;
pre_denominator = lambda_model.' * pinv(X.' * X) * lambda_model;

% Storage
t_stat_cube = zeros(nb_of_perms, 40, 40, 40);

tic;
% Routine
for ii=1:40
    for jj=1:40
        for kk=1:40

            if(data_mask(ii, jj, kk) ~= 0) %ROI
                Y_kk = big_Y(:, ii, jj, kk);
                Y_kk_perms = zeros(nb_of_perms, 16);
                % Building the permuted matrix
                for ll=1:nb_of_perms
                    for zz=1:16
                        Y_kk_perms(ll, zz) = Y_kk(D_randperm(ll, zz));
                    end
                end

                beta_vectors = pinv(X.' * X)*X.'*Y_kk_perms.';
                numerator_full = lambda_model.' * beta_vectors;
                res_vector = RX * Y_kk_perms.';
                sigma_squared_not_summed = Y_kk_perms.' .* res_vector;
                sigma_squared_vector = sum(sigma_squared_not_summed);
                denominator_full = sqrt(pre_denominator * sigma_squared_vector);
                t_stats_vector = numerator_full ./ denominator_full;
                t_stat_cube(:, ii, jj, kk) = t_stats_vector;
            end

        end
    end
end
toc;


%% Finding the maximum t-stat among all voxels for every permutation

t_stat_empirical_perms = zeros(1, nb_of_perms);

for kk=1:nb_of_perms
    t_stat_cube_slice = t_stat_cube(kk, :, :, :);
    t_stat_empirical_perms(1, kk) = max(abs(t_stat_cube_slice(:)));
end


%% Histogram

histogram(t_stat_empirical_perms, 50);
title('Empirical distribution of the maximum t-statistic');
xlabel('maximum t-statistic value');
ylabel('Number of occurences');


%% p-value

new_p_value = sum((t_stat_empirical_perms>=max_t_stat_all_voxels))/(nb_of_perms);


%% maximum t-statistic threshold

% To retrieve the histogram distribution, we need to sort our
% t_stat_empirical_perms array

t_stat_empirical_perms_sorted = sort(t_stat_empirical_perms);

% 5% of 10000 is 500 => we are looking for the index 9500
disp('The maximum t-statistic threshold value is: '+string(t_stat_empirical_perms_sorted(9500)));


% End















