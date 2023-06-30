close all;
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      COMP0114 - Coursework 4                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                              PART A                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TASK 1
%% Loading the Shepp-Logan phantom

f = phantom('Modified Shepp-Logan', 64);
cmap = colormap('hot');
imshow(f);
title('f_{true}');


%% Radon transform

%angles = [0:1:44];
angles = [0:179];
%angles = [0:4:179];
g = radon(f, angles);
imshow(g, cmap);
title('Sinogram of f_{true}');
xlabel('Angle (degrees)');
ylabel('Bins')


%% Unfiltered back-projection

bg = iradon(g, angles, 'linear', 'none', 1, size(f, 1));
% linear interpolation, no filter, no frequency rescaling, output size
imshow(bg/max(bg(:))); % rescaling ow we do not see anything
title('Unfiltered back-projection of f_{true}')


%% Filtered back-projection

bg_f = iradon(g, angles);
imshow(bg_f)
title('Filtered back-projection of f_{true}')


%% Reconstruction VS noise

% Std of the measurement noise
noise_1 = 1;
noise_2 = 2;
noise_3 = 5;
% Noisy Radon transforms
g_noisy_1 = g + normrnd(0, noise_1, size(g, 1), size(g, 2));
g_noisy_2 = g + normrnd(0, noise_2, size(g, 1), size(g, 2));
g_noisy_3 = g + normrnd(0, noise_3, size(g, 1), size(g, 2));

figure();
subplot(1, 3, 1);
%imshow(iradon(g_noisy_1, angles));
to_plot = iradon(g_noisy_1, angles, 'linear', 'none', 1, size(f, 1));
imshow(to_plot/max(to_plot(:)));
title('\sigma = '+string(noise_1));
subplot(1, 3, 2);
%imshow(iradon(g_noisy_2, angles));
to_plot = iradon(g_noisy_2, angles, 'linear', 'none', 1, size(f, 1));
imshow(to_plot/max(to_plot(:)));
title('\sigma = '+string(noise_2));
subplot(1, 3, 3);
%imshow(iradon(g_noisy_3, angles));
to_plot = iradon(g_noisy_3, angles, 'linear', 'none', 1, size(f, 1));
imshow(to_plot/max(to_plot(:)));
title('\sigma = '+string(noise_3));
sgtitle('Unfiltered noisy back-projection of f_{true}');


%% Task 1 plot

angles = [0:1:44];
g = radon(f, angles);
bg_f = iradon(g, angles);

subplot(221);
imshow(g, cmap);
title('Sinogram of f_{true}, 45 measurements from 0 to 44°');
xlabel('Angle (degrees)');
ylabel('Bins')
subplot(222);
imshow(bg_f)
title('Filtered back-projection of f_{true}, 45 measurements from 0 to 44°')

angles = [0:4:179];
g = radon(f, angles);
bg_f = iradon(g, angles);

subplot(223);
imshow(g, cmap);
title('Sinogram of f_{true}, 45 measurements from 0 to 179°');
xlabel('Angle (degrees)');
ylabel('Bins')
subplot(224);
imshow(bg_f)
title('Filtered back-projection of f_{true}, 45 measurements from 0 to 179°')


%% TASK 2
%% Creating the matrix A

%angles = [0:1:44];
angles = [0:179];
%angles = [0:4:179];
g = radon(f, angles);

A = zeros(size(g, 1)*size(g, 2), size(f, 1)*size(f, 2));

for jj=1:size(f, 2) %Column-major
    for ii=1:size(f, 1)

        % Creating the zero image with a 1 only at the pixel position
        image_jj = zeros(size(f, 1), size(f, 2));
        image_jj(ii, jj) = 1;

        % Radon transform of this image
        g_image_jj = radon(image_jj, angles);

        % Reshaping to store in the A matrix
        g_image_jj = reshape(g_image_jj, [], 1);
        A(:, jj) = g_image_jj;
    end
end

A = sparse(A);


%% Investigating the SVD

S = svds(A, 100);
%S = svd(full(A))

% Plotting
plot(S);
title('First 100 singular values of A');
xlabel('Number of the singular value')
ylabel('Singular value');


%% TASK 3
%% Preparing the data

% Noise level
theta = 0.1;

% Angles for the radon transform
angles = [0:1:179];

% We add noise to our image
%f_plus = imnoise(f, 'gaussian', 0, theta.^2); % Adding noise with theta std
%f_plus = f_plus/max(f_plus(:));
f_plus = f;

% We prepare our noisy g
g_plus = radon(f_plus, angles);
g_plus = g_plus + theta*randn(size(g_plus, 1, size(g_plus, 2)));
ATg_plus = iradon(g_plus, angles, 'linear', 'none', 1, size(f_plus, 1));
g_plus_fbp = iradon(g_plus, angles);
%ATg_plus = ATg_plus/max(ATg_plus(:)); % otherwise nothing appears!


%% Discrepancy principle

DP_alpha = fzero(@(alpha)DP(alpha, angles, f_plus, ATg_plus, g_plus, theta), 0.5);
disp('alpha found (order 0): '+string(DP_alpha));

DP_alpha = fzero(@(alpha)DPgrad(alpha, angles, f_plus, ATg_plus, g_plus, theta), 0.1);
disp('alpha found (order 1): '+string(DP_alpha));


%% Building the pcg solver

%alpha_0 = 0.10391; %for theta=0.1
%alpha_0 = 0.82882; %for theta=0.5
alpha_0 = 0.081511;
f_pcg0 = pcg(@(x)ATA(reshape(x, [], 1), alpha_0, angles, size(f_plus, 1)), reshape(ATg_plus, [], 1), 1e-06);
f_pcg0 = reshape(f_pcg0, size(f_plus, 1), size(f_plus, 2));

%alpha_1 = 0.024124; %for theta=0.1
%alpha_1 = 0.098438; %for theta=0.5
alpha_1 = 0.017495;
f_pcg1 = pcg(@(x)ATAgrad(reshape(x, [], 1), alpha_1, angles, size(f_plus, 1)), reshape(ATg_plus, [], 1), 1e-06);
f_pcg1 = reshape(f_pcg1, size(f_plus, 1), size(f_plus, 2));


%% Plotting the results

figure();
subplot(141);
imshow(g_plus_fbp);
title('Filtered back-projection');
subplot(142);
imshow(f_pcg0/max(f_pcg0(:)));
title('Regularised image (order 0)');
%xlabel('Reconstruction error: '+string(sum(sum((f_plus-f_pcg0).^2))));
xlabel('\alpha = '+string(alpha_0))
subplot(143);
imshow(f_pcg1/max(f_pcg1(:)));
title('Regularised image (order 1)');
%xlabel('Reconstruction error: '+string(sum(sum((f_plus-f_pcg1).^2))));
xlabel('\alpha = '+string(alpha_1))
subplot(144);
imshow(ATg_plus/max(ATg_plus(:)));
title('Noisy data');
sgtitle('Noise level: \theta = '+string(theta)+'; measurements: [0:1:179]')


%% TASK 4
%% Importing an image

image_test = imread('Cameraman256.png');
image_test = double(image_test)/256; % Converting to [0,1] scale
imshow(image_test);


%% Compute the Wavelet transform of the image

[a, h, v, d] = haart2(image_test);


%% Plotting the coefficients

subplot(3, 8, 1);
imshow(h{1});
ylabel('Horizontal coefficients');
title('Coefficient 1');
subplot(3, 8, 2);
imshow(h{2});
title('Coefficient 2');
subplot(3, 8, 3);
imshow(h{3});
title('Coefficient 3');
subplot(3, 8, 4);
imshow(h{4});
title('Coefficient 4');
subplot(3, 8, 5);
imshow(h{5});
title('Coefficient 5');
subplot(3, 8, 6);
imshow(h{6});
title('Coefficient 6');
subplot(3, 8, 7);
imshow(h{7});
title('Coefficient 7');
subplot(3, 8, 8);
imshow(h{8});
title('Coefficient 8');

subplot(3, 8, 9);
imshow(v{1});
ylabel('Vertical coefficients');
subplot(3, 8, 10);
imshow(v{2});
subplot(3, 8, 11);
imshow(v{3});
subplot(3, 8, 12);
imshow(v{4});
subplot(3, 8, 13);
imshow(v{5});
subplot(3, 8, 14);
imshow(v{6});
subplot(3, 8, 15);
imshow(v{7});
subplot(3, 8, 16);
imshow(v{8});

subplot(3, 8, 17);
imshow(d{1});
ylabel('Diagonal coefficients');
subplot(3, 8, 18);
imshow(d{2});
subplot(3, 8, 19);
imshow(d{3});
subplot(3, 8, 20);
imshow(d{4});
subplot(3, 8, 21);
imshow(d{5});
subplot(3, 8, 22);
imshow(d{6});
subplot(3, 8, 23);
imshow(d{7});
subplot(3, 8, 24);
imshow(d{8});


%% Comparison reconstruction VS original

% Reconstruction
image_rec = ihaart2(a, h, v, d);

% Plotting
subplot(1, 3, 1);
imshow(image_rec);
title('Reconstructed image');
subplot(1, 3, 2);
imshow(image_test);
title('Original image');
subplot(1, 3, 3);
imshow(abs(image_test-image_rec));
title('Difference between the 2');


%% Now generating noisy data

% Noise level
theta = 0.1;

image_noisy = imnoise(image_test, 'gaussian', 0, theta.^2); % Adding noise with theta std
image_noisy = image_noisy/max(image_noisy(:));

% Plotting
imshow(image_noisy);
title('Noisy image');

% Wavelet transform
[a, h, v, d] = haart2(image_noisy);


%% Thresholding

thresh_method = 's'; %'s' or 'h'
[hT4_02, vT4_02, dT4_02] = thresholdFunction(h, v, d, [1:4], 0.2, thresh_method);
[hT4_05, vT4_05, dT4_05] = thresholdFunction(h, v, d, [1:4], 0.5, thresh_method);
[hT4_07, vT4_07, dT4_07] = thresholdFunction(h, v, d, [1:4], 0.7, thresh_method);
[hT4_08, vT4_08, dT4_08] = thresholdFunction(h, v, d, [1:4], 0.8, thresh_method);
[hT4_09, vT4_09, dT4_09] = thresholdFunction(h, v, d, [1:4], 0.9, thresh_method);
[hT7_02, vT7_02, dT7_02] = thresholdFunction(h, v, d, [1:7], 0.2, thresh_method);
[hT7_05, vT7_05, dT7_05] = thresholdFunction(h, v, d, [1:7], 0.5, thresh_method);
[hT7_07, vT7_07, dT7_07] = thresholdFunction(h, v, d, [1:7], 0.7, thresh_method);
[hT7_08, vT7_08, dT7_08] = thresholdFunction(h, v, d, [1:7], 0.8, thresh_method);
[hT7_09, vT7_09, dT7_09] = thresholdFunction(h, v, d, [1:7], 0.9, thresh_method);


%% Visualizing

subplot(2, 5, 1)
imshow(ihaart2(a, hT4_02, vT4_02, dT4_02));
title('Range: 4; Percentage: 0.2');

subplot(2, 5, 2)
imshow(ihaart2(a, hT4_05, vT4_05, dT4_05));
title('Range: 4; Percentage: 0.5');

subplot(2, 5, 3)
imshow(ihaart2(a, hT4_07, vT4_07, dT4_07));
title('Range: 4; Percentage: 0.7');

subplot(2, 5, 4)
imshow(ihaart2(a, hT4_08, vT4_08, dT4_08));
title('Range: 4; Percentage: 0.8');

subplot(2, 5, 5)
imshow(ihaart2(a, hT4_09, vT4_09, dT4_09));
title('Range: 4; Percentage: 0.9');

subplot(2, 5, 6)
imshow(ihaart2(a, hT7_02, vT7_02, dT7_02));
title('Range: 7; Percentage: 0.2');

subplot(2, 5, 7)
imshow(ihaart2(a, hT7_05, vT7_05, dT7_05));
title('Range: 7; Percentage: 0.5');

subplot(2, 5, 8)
imshow(ihaart2(a, hT7_07, vT7_07, dT7_07));
title('Range: 7; Percentage: 0.7');

subplot(2, 5, 9)
imshow(ihaart2(a, hT7_08, vT7_08, dT7_08));
title('Range: 7; Percentage: 0.8');

subplot(2, 5, 10)
imshow(ihaart2(a, hT7_09, vT7_09, dT7_09));
title('Range: 7; Percentage: 0.9');



%% TASK 5
%% Preparing the data (initialization comparison)

angles = [0:179];
theta = 1;

g_tomo = radon(f, angles); % Data
g_tomo = g_tomo + theta*randn(size(g_tomo, 1, size(g_tomo, 2))); % Noisy data

f_0 = iradon(g_tomo, angles, 'linear', 'none', 1, size(f, 1));
f_0 = f_0/max(f_0(:));

f_ISTA_init = f_0; % Initialized with the unfiltered bp
% (Different init methods)
%f_ISTA_rand = abs(theta*randn(size(f, 1), size(f, 2))); % Init with noise
%f_ISTA_zero = zeros(size(f, 1), size(f, 2)); % Init with zeros


%% ISTA routine (initialization comparison)

nb_of_iter = 50;
err_plot_init = zeros(1, nb_of_iter);
err_plot_rand = zeros(1, nb_of_iter);
err_plot_zero = zeros(1, nb_of_iter);

for kk=1:nb_of_iter

f_ISTA_init = update_ISTA(f_ISTA_init, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_ISTA_init = f_ISTA_init/max(f_ISTA_init(:));
err_plot_init(kk) = norm(f-f_ISTA_init);

f_ISTA_rand = update_ISTA(f_ISTA_rand, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_ISTA_rand = f_ISTA_rand/max(f_ISTA_rand(:));
err_plot_rand(kk) = norm(f-f_ISTA_rand);

f_ISTA_zero = update_ISTA(f_ISTA_zero, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_ISTA_zero = f_ISTA_zero/max(f_ISTA_zero(:));
err_plot_zero(kk) = norm(f-f_ISTA_zero);

end

figure();
subplot(141);
imshow(f_0);
title('Noisy data');
subplot(142);
imshow(f_ISTA_init);
title('Initialized with UBP')
subplot(143);
imshow(f_ISTA_rand);
title('Initialized with noise')
subplot(144);
imshow(f_ISTA_zero);
title('Initialized with zeros')

figure();
plot(err_plot_init);
hold on;
plot(err_plot_rand)
hold on;
plot(err_plot_zero)
hold off;
legend('UBP', 'Noise', 'Zeros');
title('Reconstruction error VS original phantom');
ylabel('Pixel-wise error');
xlabel('Iteration number');


%% Preparing the data (hyperparameters comparison)

angles = [0:1:44];

g_tomo = radon(f, angles); % Data

% Adding noise and taking the UBP
f_05 = iradon(g_tomo + 0.5*randn(size(g_tomo, 1, size(g_tomo, 2))), angles, 'linear', 'none', 1, size(f, 1));
f_05 = f_05/max(f_05(:));

f_1 = iradon(g_tomo + 1*randn(size(g_tomo, 1, size(g_tomo, 2))), angles, 'linear', 'none', 1, size(f, 1));
f_1 = f_1/max(f_1(:));

f_2 = iradon(g_tomo + 2*randn(size(g_tomo, 1, size(g_tomo, 2))), angles, 'linear', 'none', 1, size(f, 1));
f_2 = f_2/max(f_2(:));


%% ISTA routine (hyperparameters comparison)

nb_of_iter = 50;

f_05_04 = f_05;
f_05_06 = f_05;
f_05_08 = f_05;

f_1_04 = f_1;
f_1_06 = f_1;
f_1_08 = f_1;

f_2_04 = f_2;
f_2_06 = f_2;
f_2_08 = f_2;

for kk=1:nb_of_iter

f_05_04 = update_ISTA(f_05_04, g_tomo, 1e-02, 0.4, angles, size(f, 1));
f_05_04 = f_05_04/max(f_05_04(:));

f_05_06 = update_ISTA(f_05_06, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_05_06 = f_05_06/max(f_05_06(:));

f_05_08 = update_ISTA(f_05_08, g_tomo, 1e-02, 0.8, angles, size(f, 1));
f_05_08 = f_05_08/max(f_05_08(:));

f_1_04 = update_ISTA(f_1_04, g_tomo, 1e-02, 0.4, angles, size(f, 1));
f_1_04 = f_1_04/max(f_1_04(:));

f_1_06 = update_ISTA(f_1_06, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_1_06 = f_1_06/max(f_1_06(:));

f_1_08 = update_ISTA(f_1_08, g_tomo, 1e-02, 0.8, angles, size(f, 1));
f_1_08 = f_1_08/max(f_1_08(:));

f_2_04 = update_ISTA(f_2_04, g_tomo, 1e-02, 0.4, angles, size(f, 1));
f_2_04 = f_2_04/max(f_2_04(:));

f_2_06 = update_ISTA(f_2_06, g_tomo, 1e-02, 0.6, angles, size(f, 1));
f_2_06 = f_2_06/max(f_2_06(:));

f_2_08 = update_ISTA(f_2_08, g_tomo, 1e-02, 0.8, angles, size(f, 1));
f_2_08 = f_2_08/max(f_2_08(:));

end


%% Plotting

figure();
subplot(331);
imshow(f_05_04);
title('\mu = 0.4')
ylabel('\theta = 0.5')
subplot(332);
imshow(f_05_06);
title('\mu = 0.6')
subplot(333);
imshow(f_05_08);
title('\mu = 0.8')
subplot(334);
imshow(f_1_04);
ylabel('\theta = 1')
subplot(335);
imshow(f_1_06);
subplot(336);
imshow(f_1_08);
subplot(337);
imshow(f_2_04);
ylabel('\theta = 2')
subplot(338);
imshow(f_2_06);
subplot(339);
imshow(f_2_08);


%% APPENDIX

% Krylov solver, 0-order regularization

function z = ATA(f, alpha, angles, output_size)
    f = reshape(f, output_size, output_size);
    y = radon(f, angles);
    z = iradon(y, angles, 'linear', 'none', 1, output_size) + alpha*f;
    z = reshape(z, [], 1);
end

% Krylov solver,  1st-order regularization

function z = ATAgrad(f, alpha, angles, output_size)
    f = reshape(f, output_size, output_size);
    y = radon(f, angles);
    [Gmag, ~] = imgradient(f);
    z = iradon(y, angles, 'linear', 'none', 1, output_size) + alpha*(Gmag);
    z = reshape(z, [], 1);
end

% Discrepancy principle

function res_DP = DP(alpha, angles, f_plus, ATg_plus, g, theta)

    g = reshape(g, [], 1);

    % Computing f_alpha with pcg
    [fa_pcg, ~] = pcg(@(x)ATA(reshape(x, [], 1), alpha, angles, size(f_plus, 1)), reshape(ATg_plus, [], 1), 1e-06);

    % Taking the Radon transform of fa_pcg before using it in r_alpha
    fa_pcg = reshape(fa_pcg, size(f_plus, 1), size(f_plus, 2));
    Afa_pcg = radon(fa_pcg, angles);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end

function res_DP = DPgrad(alpha, angles, f_plus, ATg_plus, g, theta)
    
    g = reshape(g, [], 1);

    % Computing f_alpha with pcg
    [fa_pcg, ~] = pcg(@(x)ATAgrad(reshape(x, [], 1), alpha, angles, size(f_plus, 1)), reshape(ATg_plus, [], 1), 1e-06);

    % Taking the Radon transform of fa_pcg before using it in r_alpha
    fa_pcg = reshape(fa_pcg, size(f_plus, 1), size(f_plus, 2));
    Afa_pcg = radon(fa_pcg, angles);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end

% Threshold function

function [hT, vT, dT] = thresholdFunction(h, v, d, range, percentage, thresholdingType)
    range_real = size(h, 2);
    
    % First, let's find our thresholding value
    % Putting all the coefficients in one big vector
    allCoeffs = [];
    for kk=1:range(end)
        allCoeffs = [allCoeffs abs(reshape(h{kk}, 1, [])) abs(reshape(v{kk}, 1, [])) abs(reshape(d{kk}, 1, []))];
    end
    
    % Sorting that vector
    allCoeffs = sort(allCoeffs);
    
    % Finding our thresholding value
    tIdx = min(round(percentage*size(allCoeffs, 2)), size(allCoeffs, 2));
    tVal = allCoeffs(1, tIdx);
    %disp(tVal);
    
    % Thresholding
    hT = cell(1, range_real);
    vT = cell(1, range_real);
    dT = cell(1, range_real);
    for kk=1:range(end)
        hT{kk} = wthresh(h{kk}, thresholdingType, tVal);
        vT{kk} = wthresh(v{kk}, thresholdingType, tVal);
        dT{kk} = wthresh(d{kk}, thresholdingType, tVal);
    end
    % Adding the non-thresholded coefficients
    for kk=range(end)+1:range_real
        hT{kk} = h{kk};
        vT{kk} = v{kk};
        dT{kk} = d{kk};
    end

end


function f_next_step = update_ISTA(f_previous_step, g, lambda, percentage, angles, output_size)

    % Computing what we want to input in our thresholding function
    y = radon(f_previous_step, angles) - g;
    %input_ISTA = f_previous_step - lambda * iradon(y, angles, 'linear', 'none', 1, output_size)/100;
    input_ISTA = f_previous_step - lambda * iradon(y, angles, 'linear', 'none', 1, output_size);

    % Applying the thresholding function
    % Getting the wavelet coefficients
    [a, h, v, d] = haart2(input_ISTA);

    % Denoising by percentage
    %[hT, vT, dT] = thresholdFunction(h, v, d, [1:7], lambda*alpha, 'h'); %If we want to keep alpha
    [hT, vT, dT] = thresholdFunction(h, v, d, [1:4], percentage, 's');
    
    % Reconstruction
    f_next_step = abs(ihaart2(a, hT, vT, dT));

end


% End