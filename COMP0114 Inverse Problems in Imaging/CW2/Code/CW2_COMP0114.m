close all;
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      Coursework 2 - Week 1                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% a) Importing an image

img = double(imread("Cameraman256.png")); % Converting to double
img = img/max(img(:)); % Setting the values in [0,1]
gray_map = colormap('gray'); % Colormap


%% a) Visualizing

figure();
imshow(img);


%% b) Convolution mapping/Gaussian kernel

% We use directly the imgaussfilt function from MATLAB
sigma = 1;
filteredImg = imgaussfilt(img, sigma);


%% b) Visualizing

figure();
subplot(121);
imshow(img);
title('Original image')
subplot(122);
imshow(filteredImg)
title('Blurred image')


%% c) Defining parameters for pcg

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g(g<0) = 0;


ATg = imgaussfilt(g, sigma);

%g = reshape(g, []);
ATg = reshape(ATg, [], 1);


%% c) Applying pcg

alpha = 2;
fa_pcg = pcg(@(x)ATA(x, alpha, sigma), ATg, 1e-10, 1);
fa_pcg = reshape(fa_pcg, 256, 256);

fa_gmres = gmres(@(x)ATA(x, alpha, sigma), ATg, [], 1e-10, 1);
fa_gmres = reshape(fa_gmres, 256, 256);

%fa_lsqr = lsqr(@(x, transposeFlag)ATA_with_transp(x, alpha, sigma, transposeFlag), ATg, 1e-10);
%fa_lsqr = lsqr(@(x)ATA(x, alpha, sigma), ATg);
%fa_lsqr = reshape(fa_lsqr, 256, 256);


%% c) Visualizing

figure();
subplot(221);
imshow(img);
title('Ground truth');
subplot(222);
imshow(g);
title("Blurred+noisy image");
subplot(223);
%imshow(fa_pcg);
imshow(fa_pcg/max(fa_pcg(:)));
title('Reconstruction PCG');
xlabel('Estimation error: '+string(norm(img-fa_pcg)))
subplot(224);
%imshow(fa_gmres);
imshow(fa_gmres/max(fa_gmres(:)));
title('Reconstruction GMRES');
xlabel('Estimation error: '+string(norm(img-fa_gmres)))


%% d) Parameters for lsqr

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

%ATg = imgaussfilt(g, sigma);
gaug = [g; zeros(size(g))];

%g = reshape(g, []);
gaug = reshape(gaug, [], 1);


%% d) Applying lsqr

alpha = 1;
faaug_lsqr = lsqr(@(x, transposeFlag)Aaug(x, alpha, sigma, transposeFlag), gaug, 1e-10, 1);
%size(faaug_lsqr)
faaug_lsqr = reshape(faaug_lsqr(1:65536), 256, 256);

%%

figure();
subplot(221);
imshow(g);
title("Blurred+noisy image");
subplot(222);
%imshow(fa_pcg);
imshow(fa_pcg/max(fa_pcg(:)));
title('PCG-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_pcg)))
subplot(223);
%imshow(fa_gmres);
imshow(fa_gmres/max(fa_gmres(:)));
title('GMRES-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_pcg)))
subplot(224);
%imshow(fa_lsqr);
imshow(faaug_lsqr/max(faaug_lsqr(:)));
title('LSQR-reconstructed augmented image');
xlabel('Estimation error: '+string(norm(img-faaug_lsqr)))




%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      Coursework 2 - Week 2                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2) Discrepency Principle initialization

% PCG VERSION

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
% We have obtained g tilde (flattened): for r_alpha comparison
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% We have obtained ATg (flattened): for f_alpha computation


% LSQR VERSION
%{
sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

%ATg = imgaussfilt(g, sigma);
gaug = [g; zeros(size(g))];

%g = reshape(g, []);
gaug = reshape(gaug, [], 1);
%}


%% 2) DP running

% Using fzero to find the alphas
DP_alpha = fzero(@(alpha)DP(alpha, sigma, theta, g, ATg), 0.5);
%DP_alpha = fzero(@(alpha)DP(alpha, sigma, theta, gaug, g), 1);
disp('alpha found: '+string(DP_alpha));


%% DP tests on sigma

sigmas = linspace(0.05, 1);
alphas_result_wrt_sigma = zeros(length(sigmas));
for tt=1:length(sigmas)
    sigma = sigmas(tt);
    theta = 0.1;
    g = imgaussfilt(img, sigma);
    g = imnoise(g, 'gaussian', 0, theta.^2);
    g = reshape(g, [], 1);
    ATg = imgaussfilt(g, sigma);
    ATg = reshape(ATg, [], 1);
    DP_alpha = fzero(@(alpha)DP(alpha, sigma, g, ATg), 0.5);
    alphas_result_wrt_sigma(tt) = DP_alpha;
end


%% DP tests on sigma plotting

figure()
plot(sigmas, alphas_result_wrt_sigma);


%% DP tests on theta

%We choose sigma = 1 bc seems ok
thetas = linspace(0.01, 1);
alphas_result_wrt_theta = zeros(length(thetas));
for tt=1:length(thetas)
    sigma = 0.25;
    theta = thetas(tt);
    g = imgaussfilt(img, sigma);
    g = imnoise(g, 'gaussian', 0, theta.^2);
    g = reshape(g, [], 1);
    ATg = imgaussfilt(g, sigma);
    ATg = reshape(ATg, [], 1);
    DP_alpha = fzero(@(alpha)DP(alpha, sigma, g, ATg), 1);
    alphas_result_wrt_theta(tt) = DP_alpha;
end


%% DP tests on theta on plotting

figure()
plot(thetas, alphas_result_wrt_theta);



%% 2) L-curve 

alphas = linspace(0.5, 50, 200);
lcurve_x = zeros(200, 1);
lcurve_y = zeros(200, 1);

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
% We have obtained g tilde (flattened): for r_alpha comparison
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% We have obtained ATg (flattened): for f_alpha computation
gaug = [g; zeros(size(g))];
gaug = reshape(gaug, [], 1);

for tt=1:200
    [res_DP, res_reg] = L_curve(alphas(tt), sigma, theta, g, ATg);

    lcurve_x(tt) = res_DP;
    lcurve_y(tt) = res_reg;
end


%% 2) L-curve plotting

loglog(lcurve_x, lcurve_y);
title('L-curve for the Ridge regularizer');
xlabel('Norm squared of the residual');
ylabel('Norm squared of the regularizer')
%loglog(lcurve_y, lcurve_x);
%plot(lcurve_x, lcurve_y);


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      Coursework 2 - Week 3                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% a) Gradient operator

n = 256;
e = ones(n,1);
D = spdiags([e -2*e e],-1:1,n,n);


%% b) Solver for Gradient operator

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

ATg = imgaussfilt(g, sigma);

%g = reshape(g, []);
ATg = reshape(ATg, [], 1);

alpha = 0.074625;
fa_pcg = pcg(@(x)ATAgrad(x, alpha, sigma), ATg, 1e-10);
fa_pcg = reshape(fa_pcg, 256, 256);

fa_gmres = gmres(@(x)ATAgrad(x, alpha, sigma), ATg, [], 1e-10);
fa_gmres = reshape(fa_gmres, 256, 256);

%% b) Visualizing

figure();
subplot(221);
imshow(img);
title('Ground truth');
subplot(222);
imshow(g);
title("Blurred+noisy image");
subplot(223);
%imshow(fa_pcg);
imshow(fa_pcg/max(fa_pcg(:)));
title('Reconstruction PCG');
xlabel('Estimation error: '+string(norm(img-fa_pcg)))
subplot(224);
%imshow(fa_gmres);
imshow(fa_gmres/max(fa_gmres(:)));
title('Reconstruction GMRES');
xlabel('Estimation error: '+string(norm(img-fa_gmres)))

%% b) lsqr

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

%ATg = imgaussfilt(g, sigma);
gaug = [g; zeros(size(g))];

%g = reshape(g, []);
gaug = reshape(gaug, [], 1);

faaug_lsqr = lsqr(@(x, transposeFlag)Aauggrad(x, alpha, sigma, transposeFlag), gaug, 1e-10);
faaug_lsqr = reshape(faaug_lsqr(1:65536), 256, 256);


%% b) Plotting with lsqr

figure();
subplot(221);
imshow(g);
title("Blurred+noisy image");
subplot(222);
%imshow(fa_pcg);
imshow(fa_pcg/max(fa_pcg(:)));
title('PCG-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_pcg)))
subplot(223);
%imshow(fa_gmres);
imshow(fa_gmres/max(fa_gmres(:)));
title('GMRES-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_gmres)))
subplot(224);
%imshow(faaug_lsqr);
imshow(faaug_lsqr/max(faaug_lsqr(:)));
title('LSQR-reconstructed augmented image')
xlabel('Estimation error: '+string(norm(img-faaug_lsqr)))

%% Choice of alpha for Derivative reg

% Simple DP
% Initialization
sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% Computing
DP_alpha = fzero(@(alpha)DPgrad(alpha, sigma, theta, g, ATg), 0.01);
disp('alpha found: '+string(DP_alpha));


%% L-curve for gradient reg

alphas = linspace(0.005, 1, 200);
lcurve_x = zeros(200, 1);
lcurve_y = zeros(200, 1);

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
% We have obtained g tilde (flattened): for r_alpha comparison
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% We have obtained ATg (flattened): for f_alpha computation
gaug = [g; zeros(size(g))];
gaug = reshape(gaug, [], 1);

for tt=1:200
    [res_DP, res_reg] = L_curvegrad(alphas(tt), sigma, theta, g, ATg);

    lcurve_x(tt) = res_DP;
    lcurve_y(tt) = res_reg;
end

%% Plotting the L-curve for gradient reg

figure();
loglog(lcurve_x, lcurve_y);
title('L-curve for the spatial derivative regularizer');
xlabel('Norm squared of the residual');
ylabel('Norm squared of the regularizer')


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      Coursework 2 - Week 4                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%

n = 256*256;
e = ones(n,1);
Lapl = spdiags([e -4*e e],-1:1,n,n);

%%

imshow(full(A(1:500,1:500)));


%% b) Solver for anisotropic operator

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

ATg = imgaussfilt(g, sigma);

%g = reshape(g, []);
ATg = reshape(ATg, [], 1);

alpha = 1.4975;
%figure();
fa_pcg_2 = pcg(@(x)ATAanisotropic(x, alpha, sigma), ATg, 1e-10, 100, [], [], reshape(fa_pcg, [], 1));
fa_pcg_2 = reshape(fa_pcg_2, 256, 256); 

%figure();
fa_gmres_2 = gmres(@(x)ATAanisotropic(x, alpha, sigma), ATg, [], 1e-10, 100, [], [], reshape(fa_gmres, [], 1));
fa_gmres_2 = reshape(fa_gmres_2, 256, 256);


%% b) Visualizing

figure();
subplot(221);
imshow(img);
title('Ground truth');
subplot(222);
imshow(g);
title("Blurred+noisy image");
subplot(223);
%imshow(fa_pcg_2);
imshow(fa_pcg_2/max(fa_pcg_2(:)));
title('Reconstruction PCG');
xlabel('Estimation error: '+string(norm(img-fa_pcg_2)))
subplot(224);
%imshow(fa_gmres_2);
imshow(fa_gmres_2/max(fa_gmres_2(:)));
title('Reconstruction GMRES');
xlabel('Estimation error: '+string(norm(img-fa_gmres_2)))

%% b) lsqr

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

%ATg = imgaussfilt(g, sigma);
gaug = [g; zeros(size(g))];

%g = reshape(g, []);
gaug = reshape(gaug, [], 1);

faaug_lsqr_2 = lsqr(@(x, transposeFlag)Aauganisotropic(x, alpha, sigma, transposeFlag), gaug, 1e-10, 100, [], [], reshape([faaug_lsqr; faaug_lsqr], [], 1)); %
faaug_lsqr_2 = reshape(faaug_lsqr_2(1:65536), 256, 256);


%% b) Plotting with lsqr

figure();
subplot(221);
imshow(g);
title("Blurred+noisy image");
subplot(222);
%imshow(fa_pcg_2);
imshow(fa_pcg_2/max(fa_pcg_2(:)));
title('PCG-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_pcg_2)))
subplot(223);
%imshow(fa_gmres_2);
imshow(fa_gmres_2/max(fa_gmres_2(:)));
title('GMRES-reconstructed image');
xlabel('Estimation error: '+string(norm(img-fa_gmres_2)))
subplot(224);
%imshow(faaug_lsqr_2);
imshow(faaug_lsqr_2/max(faaug_lsqr_2(:)));
title('LSQR-reconstructed augmented image')
xlabel('Estimation error: '+string(norm(img-faaug_lsqr_2)))

%% Choice of alpha for anisotropic reg

% Simple DP
% Initialization
sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% Computing
DP_alpha = fzero(@(alpha)DPanisotropic(alpha, sigma, theta, g, ATg, fa_pcg), 4);
disp('alpha found: '+string(DP_alpha));


%% L-curve for anisotropic reg

alphas = linspace(0.0005, 2, 300);
lcurve_x = zeros(200, 1);
lcurve_y = zeros(200, 1);

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std
g = reshape(g, [], 1);
% We have obtained g tilde (flattened): for r_alpha comparison
ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);
% We have obtained ATg (flattened): for f_alpha computation
gaug = [g; zeros(size(g))];
gaug = reshape(gaug, [], 1);

for tt=1:300
    [res_DP, res_reg] = L_curveanisotropic(alphas(tt), sigma, theta, g, ATg, fa_pcg);

    lcurve_x(tt) = res_DP;
    lcurve_y(tt) = res_reg;
end

%% Plotting the L-curve for anisotropic reg

figure();
loglog(lcurve_x, lcurve_y);
title('L-curve for the spatial derivative regularizer');
xlabel('Norm squared of the residual');
ylabel('Norm squared of the regularizer')


%% ITERATIVE METHOD (with plot every iteration)

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);

alpha = 1.4759;

figure();
img_ii = g;
for ii=1:100
    diffusivity = perona_malik(img_ii);
    if(ii==1)
    %img_ii = pcg(@(x)ATAanisotropic(x, alpha, sigma), reshape(img,[],1), 1e-10, 100, [], [], reshape(fa_pcg, [], 1));
    img_ii = pcg(@(x)ATAgrad(x, alpha, sigma), reshape(img,[],1), 1e-10, 100);
    else
    img_ii = pcg(@(x)ATAanisotropic(x, alpha, sigma), reshape(img,[],1), 1e-10, 100, [], [], reshape(img_ii, [], 1));    
    %img_ii = pcg(@(x)ATAgrad(x, alpha, sigma), reshape(img,[],1), 1e-10, 100);    
    end
    img_ii = reshape(img_ii, 256, 256); 
    subplot(131);
    imshow(diffusivity);
    xlabel('Diffusivity')
    subplot(132);
    imshow(img_ii);
    xlabel('Unblurred image')
    subplot(133);
    imshow(img);
    xlabel('Ground truth')
    sgtitle('Estimation error: '+string(norm(img-img_ii))+'. Iteration: '+string(ii)+'.');
    pause(1);
end
    
%% ITERATIVE METHOD (with plot only for the first 3 iterations)

sigma = 1;
theta = 0.05;
g = imgaussfilt(img, sigma); % Initial blurring
g = imnoise(g, 'gaussian', 0, theta.^2); % Adding noise with theta std

ATg = imgaussfilt(g, sigma);
ATg = reshape(ATg, [], 1);

alpha = 1.4759;

figure();
img_ii = g;
for ii=1:5
    diffusivity = perona_malik(img_ii);
    if(ii==1 || ii==3)
    img_ii = pcg(@(x)ATAanisotropic(x, alpha, sigma), reshape(img,[],1), 1e-10, 100, [], [], reshape(fa_pcg, [], 1));
    %img_ii = pcg(@(x)ATAgrad(x, alpha, sigma), reshape(img,[],1), 1e-10, 100);
    else
    %img_ii = pcg(@(x)ATAanisotropic(x, alpha, sigma), reshape(img,[],1), 1e-10, 100, [], [], reshape(img_ii, [], 1));    
    img_ii = pcg(@(x)ATAgrad(x, alpha, sigma), reshape(img,[],1), 1e-10, 100);    
    end
    img_ii = reshape(img_ii, 256, 256); 

    if(ii==1)
        subplot(331);
        imshow(diffusivity);
        ylabel({'Estimation error: '+string(norm(img-img_ii))+'.','Iteration: '+string(ii)+'.'});
        subplot(332);
        imshow(img_ii);
        subplot(333);
        imshow(img);
        %sgtitle('Estimation error: '+string(norm(img-img_ii))+'. Iteration: '+string(ii)+'.');
    end
    if(ii==2)
        subplot(334);
        imshow(diffusivity);
        ylabel({'Estimation error: '+string(norm(img-img_ii))+'.','Iteration: '+string(ii)+'.'});
        subplot(335);
        imshow(img_ii);
        subplot(336);
        imshow(img);
        %sgtitle('Estimation error: '+string(norm(img-img_ii))+'. Iteration: '+string(ii)+'.');
    end
    if(ii==3)
        subplot(337);
        imshow(diffusivity);
        ylabel({'Estimation error: '+string(norm(img-img_ii))+'.','Iteration: '+string(ii)+'.'});
        xlabel('Diffusivity')
        subplot(338);
        imshow(img_ii);
        xlabel('Unblurred image')
        subplot(339);
        imshow(img);
        xlabel('Ground truth')
        %sgtitle('Estimation error: '+string(norm(img-img_ii))+'. Iteration: '+string(ii)+'.');
    end
    
end



%% Works Laplacian!

img_ii = img;
img_ii_temoin = img;
for ii=1:50
    gamma = perona_malik(img_ii);
    img_ii = img_ii - 0.25*fulldiff2(img_ii).*gamma;
    img_ii_temoin = img_ii_temoin - 0.25*fulldiff2(img_ii_temoin);
    if(ii==5)
        img_ii_5 = img_ii;
        img_ii_temoin_5 = img_ii_temoin;
    end
    if(ii==10)
        img_ii_10 = img_ii;
        img_ii_temoin_10 = img_ii_temoin;
    end
    if(ii==15)
        img_ii_15 = img_ii;
        img_ii_temoin_15 = img_ii_temoin;
    end
    pause(0.05);
end

figure();
subplot(321);
imshow(img_ii_5);
title("Edge Laplacian, i=5");
subplot(322);
imshow(img_ii_temoin_5);
title("Laplacian, i=5");
subplot(323);
imshow(img_ii_10);
title("Edge Laplacian, i=10");
subplot(324);
imshow(img_ii_temoin_10);
title("Laplacian, i=10");
subplot(325);
imshow(img_ii_15);
title("Edge Laplacian, i=15");
subplot(326);
imshow(img_ii_temoin_15);
title("Laplacian, i=15");


%%

figure();
subplot(221);
imshow(perona_malik(img, 0.05));
title('Percentage of normalization: 5%');
subplot(222);
imshow(perona_malik(img, 0.1));
title('Percentage of normalization: 10%');
subplot(223);
imshow(perona_malik(img, 0.15));
title('Percentage of normalization: 15%');
subplot(224);
imshow(perona_malik(img, 0.2));
title('Percentage of normalization: 20%');


%% Appendix - functions

function gamma = perona_malik(f)
    Df = fulldiffnorm(f);
    T = 0.2*max(Df(:));
    %T = 0.1*max(Df(:));
    %T = k*max(Df(:));
    gamma = exp(-Df/T);
    %gamma = exp(-Df/T)+normrnd(0,0.25,256,256);
end

function Df = fulldiff(f)
    deriv_x = diff(f,1,1); %255*256
    deriv_y = diff(f,1,2); %256*255
    
    deriv_x_padded = zeros(256,256);
    %deriv_x_padded(2:end, :) = deriv_x;
    %deriv_x_padded(1, :) = deriv_x_padded(2, :);
    deriv_x_padded(1:end-1, :) = deriv_x;
    deriv_x_padded(end, :) = deriv_x_padded(end-1, :);
    
    deriv_y_padded = zeros(256,256);
    %deriv_y_padded(:, 2:end) = deriv_y;
    %deriv_y_padded(:, 1) = deriv_y_padded(:, 2);
    deriv_y_padded(:, 1:end-1) = deriv_y;
    deriv_y_padded(:, end) = deriv_y_padded(:, end-1);

    Df = deriv_x_padded + deriv_y_padded;
end

function Df2 = fulldiff2(f)
    deriv_x = diff(f,1,1); %255*256
    deriv_y = diff(f,1,2); %256*255
    
    deriv_x_padded = zeros(256,256);
    deriv_x_padded(1:end-1, :) = deriv_x;
    deriv_x_padded(end, :) = deriv_x_padded(end-1, :);
    deriv_x2 = diff(deriv_x_padded,1,1);
    deriv_x2_padded = zeros(256,256);
    deriv_x2_padded(2:end, :) = deriv_x2;
    deriv_x2_padded(1, :) = deriv_x2_padded(2, :);

    deriv_y_padded = zeros(256,256);
    deriv_y_padded(:, 1:end-1) = deriv_y;
    deriv_y_padded(:, end) = deriv_y_padded(:, end-1);
    deriv_y2 = diff(deriv_y_padded,1,2);
    deriv_y2_padded = zeros(256,256);
    deriv_y2_padded(:, 2:end) = deriv_y2;
    deriv_y2_padded(:, 1) = deriv_y2_padded(:, 2);

    Df2 = -(deriv_x2_padded + deriv_y2_padded);
end

function Df = diffx(f)
    deriv_x = diff(f,1,1); %255*256
    
    a = zeros(256,256);
    a(2:end, :) = deriv_x;
    a(1, :) = a(2, :);
    
    Df = a;
end

function Df = diffx2(f)
    deriv_x = diff(f,1,1); %255*256
    
    a = zeros(256,256);
    a(2:end, :) = deriv_x;
    a(1, :) = a(2, :);
    a2 = zeros(256,256);
    deriv_x2 = diff(a,1,1);
    a2(1:end-1, :) = deriv_x2;
    a2(end, :) = a2(end-1, :);
    
    Df = a2;
end

function Df = diffy(f)
    deriv_y = diff(f,1,2); %256*255
    
    b = zeros(256,256);
    b(:, 2:end) = deriv_y;
    b(:, 1) = b(:, 2);

    Df = b;
end

function Df = diffy2(f)
    deriv_y = diff(f,1,2); %256*255
    
    b = zeros(256,256);
    b(:, 2:end) = deriv_y;
    b(:, 1) = b(:, 2);
    b2 = zeros(256,256);
    deriv_y2 = diff(b,1,2);
    b2(:, 1:end-1) = deriv_y2;
    b2(:, end) = b2(:, end-1);

    Df = b2;
end

function Df = fulldiffnorm(f)
    deriv_x = diff(f,1,1); %255*256
    deriv_y = diff(f,1,2); %256*255
    
    deriv_x_padded = zeros(256,256);
    deriv_x_padded(2:end, :) = deriv_x;
    deriv_x_padded(1, :) = deriv_x_padded(2, :);
    deriv_x_padded2 = deriv_x_padded.^2;
    
    deriv_y_padded = zeros(256,256);
    deriv_y_padded(:, 2:end) = deriv_y;
    deriv_y_padded(:, 1) = deriv_x_padded(:, 2);
    deriv_y_padded2 = deriv_y_padded.^2;

    Df = sqrt(deriv_x_padded2+deriv_y_padded2);
end

%{
function z = ATAanisotropic(f, alpha, sigma)
    f = reshape(f, 256, 256);
    y = imgaussfilt(f, sigma);
    gamma = perona_malik(f);
    %z = imgaussfilt(y, sigma) + alpha*gamma.*reshape(Lapl*reshape(f, [], 1), 256, 256);
    %gamma = ones(256,256);
    z = imgaussfilt(y, sigma) - alpha*times(gamma,fulldiff2(f)); %+abs(normrnd(0,1,256,256));
    %diffxf = diffx(f);
    %gammadiffxf = gamma.*diffxf;
    %diffygammadiffxf = diffy(gammadiffxf);
    %z = imgaussfilt(y, sigma) + alpha*diffygammadiffxf; %times(gamma,fulldiff(f));
    %imshow(z);
    z = reshape(z, [], 1);
end
%}

function z = ATAanisotropic(f, alpha, sigma)
    f = reshape(f, 256, 256);
    y = imgaussfilt(f, sigma);
    gamma = perona_malik(f);
    z = imgaussfilt(y, sigma) - alpha*gamma.*fulldiff2(f);
    z = reshape(z, [], 1);
end

function z = ATAgrad(f, alpha, sigma)
    f = reshape(f, 256, 256);
    y = imgaussfilt(f, sigma);
    z = imgaussfilt(y, sigma) + alpha.*fulldiff(f);
    z = reshape(z, [], 1);
end

function z = ATA(f, alpha, sigma)
    f = reshape(f, 256, 256);
    y = imgaussfilt(f, sigma);
    z = imgaussfilt(y, sigma) + alpha.*f;
    z = reshape(z, [], 1);
end


%{
function z = ATA_with_transp(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        f = reshape(f, 256, 256);
        y = imgaussfilt(f, sigma);
        z = imgaussfilt(y, sigma) + alpha.*f;
        z = reshape(z, [], 1);
    case 'transp'
        f = reshape(f, 256, 256);
        y = imgaussfilt(f, sigma);
        z = imgaussfilt(y, sigma) + alpha.*f;
        z = reshape(z, [], 1);
end
end
%}

%{
function z=Aaug(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        % implementation of the augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*f;
        z = [y; y_alpha];
        z = reshape(z, [], 1);
    case 'transp'
        % implementation of the transposed augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*f;
        z = [y y_alpha];
        z = reshape(z, [], 1);
    otherwise
        error('input transposeFlag has to be ''transp'' or ''notransp''')
end
end
%}
%{
function z=Aaug(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        % implementation of the augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*f;
        z = [y; y_alpha];
        z = reshape(z, [], 1);
    case 'transp'
        % implementation of the transposed augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*f;
        z = [y y_alpha];
        z = reshape(z, [], 1);
    otherwise
        error('input transposeFlag has to be ''transp'' or ''notransp''')
end
end
%}


function z=Aaug(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        % implementation of the augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*f;
        z = [y; y_alpha];
        z = reshape(z, [], 1);
    case 'transp'
        % implementation of the transposed augmented matrix multiplication
        f = reshape(f, 2*256, 256);
        y = imgaussfilt(f(1:256, :), sigma);
        y_alpha = sqrt(alpha).*f(257:end, :);
        z = [y y_alpha];
        z = reshape(z, [], 1);
    otherwise
        error('input transposeFlag has to be ''transp'' or ''notransp''')
end
end

function z=Aauggrad(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        % implementation of the augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        y_alpha = sqrt(alpha).*fulldiff(f);
        z = [y; y_alpha];
        z = reshape(z, [], 1);
    case 'transp'
        % implementation of the transposed augmented matrix multiplication
        f = reshape(f, 2*256, 256);
        y = imgaussfilt(f(1:256, :), sigma);
        y_alpha = sqrt(alpha).*fulldiff(f(257:end, :));
        z = [y y_alpha];
        z = reshape(z, [], 1);
    otherwise
        error('input transposeFlag has to be ''transp'' or ''notransp''')
end
end

function z=Aauganisotropic(f, alpha, sigma, transposeFlag)
switch transposeFlag
    case 'notransp'
        % implementation of the augmented matrix multiplication
        f = reshape(f(1:65536), 256, 256);
        y = imgaussfilt(f, sigma);
        gamma = perona_malik(f);
        y_alpha = sqrt(alpha).*gamma.*fulldiff2(f);
        z = [y; y_alpha];
        z = reshape(z, [], 1);
    case 'transp'
        % implementation of the transposed augmented matrix multiplication
        f = reshape(f, 2*256, 256);
        y = imgaussfilt(f(1:256, :), sigma);
        gamma = perona_malik(f(257:end, :));
        y_alpha = sqrt(alpha).*gamma.*fulldiff2(f(257:end, :));
        z = [y y_alpha];
        z = reshape(z, [], 1);
    otherwise
        error('input transposeFlag has to be ''transp'' or ''notransp''')
end
end


function res_DP = DP(alpha, sigma, theta, g, ATg)
    
    % Computing f_alpha with least squares
    fa_pcg = pcg(@(x)ATA(x, alpha, sigma), ATg, 1e-10);

    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg = reshape(fa_pcg, 256, 256);
    Afa_pcg = imgaussfilt(fa_pcg, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end

function res_DP = DPgrad(alpha, sigma, theta, g, ATg)
    
    % Computing f_alpha with least squares
    fa_pcg = pcg(@(x)ATAgrad(x, alpha, sigma), ATg, 1e-10);

    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg = reshape(fa_pcg, 256, 256);
    Afa_pcg = imgaussfilt(fa_pcg, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end

function res_DP = DPanisotropic(alpha, sigma, theta, g, ATg, estimate_pcg)
    
    % Computing f_alpha with least squares
    fa_pcg_2 = pcg(@(x)ATAanisotropic(x, alpha, sigma), ATg, 1e-10, 100, [], [], reshape(estimate_pcg, [], 1));
    
    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg_2 = reshape(fa_pcg_2, 256, 256); 
    Afa_pcg = imgaussfilt(fa_pcg_2, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end



%{
function res_DP = DP(alpha, sigma, theta, gaug, g)
    
    % Computing f_alpha with least squares
    faaug_lsqr = lsqr(@(x, transposeFlag)Aaug(x, alpha, sigma, transposeFlag), gaug, 1e-10,50);
    
    % Blurring fa_lsqr before using it in r_alpha
    faaug_lsqr = reshape(faaug_lsqr(1:65536), 256, 256);
    Afaaug_lsqr = imgaussfilt(faaug_lsqr, sigma);
    Afaaug_lsqr = reshape(Afaaug_lsqr, [], 1);

    % Computing r_alpha
    r_alpha = g - Afaaug_lsqr;
    n = length(r_alpha);

    % DP equation
    res_DP = (1/n)*(norm(r_alpha).^2) - theta.^2;

end
%}


function [res_DP, res_reg] = L_curve(alpha, sigma, theta, g, ATg)
    
    % Computing f_alpha with least squares
    fa_pcg = pcg(@(x)ATA(x, alpha, sigma), ATg, 1e-10);

    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg = reshape(fa_pcg, 256, 256);
    Afa_pcg = imgaussfilt(fa_pcg, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;

    % L-curve elements
    res_DP = (norm(r_alpha, 2).^2);
    res_reg = (norm(fa_pcg, 2).^2);


end

function [res_DP, res_reg] = L_curvegrad(alpha, sigma, theta, g, ATg)
    
    % Computing f_alpha with least squares
    fa_pcg = pcg(@(x)ATAgrad(x, alpha, sigma), ATg, 1e-10);

    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg = reshape(fa_pcg, 256, 256);
    Afa_pcg = imgaussfilt(fa_pcg, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;

    % L-curve elements
    res_DP = (norm(r_alpha, 2).^2);
    res_reg = (norm(fulldiff2(fa_pcg), 2).^2);


end

function [res_DP, res_reg] = L_curveanisotropic(alpha, sigma, theta, g, ATg, estimate_pcg)
    
    % Computing f_alpha with least squares
    fa_pcg_2 = pcg(@(x)ATAanisotropic(x, alpha, sigma), ATg, 1e-10, 100, [], [], reshape(estimate_pcg, [], 1));
    
    % Blurring fa_lsqr before using it in r_alpha
    fa_pcg_2 = reshape(fa_pcg_2, 256, 256); 
    Afa_pcg = imgaussfilt(fa_pcg_2, sigma);
    Afa_pcg = reshape(Afa_pcg, [], 1);

    % Computing r_alpha
    r_alpha = g - Afa_pcg;

    % L-curve elements
    res_DP = (norm(r_alpha, 2).^2);
    res_reg = (norm(sqrt(perona_malik(fa_pcg_2)).*fulldiff2(fa_pcg_2), 2).^2);


end


%{
function [res_DP, res_reg] = L_curve(alpha, sigma, theta, g, gaug)
    
    faaug_lsqr = lsqr(@(x, transposeFlag)Aaug(x, alpha, sigma, transposeFlag), gaug, 1e-10, 20);
    faaug_lsqr = reshape(faaug_lsqr(1:65536), 256, 256);

    % Blurring fa_lsqr before using it in r_alpha
    f_a = faaug_lsqr;
    Afaaug_lsqr = imgaussfilt(faaug_lsqr, sigma);
    Afaaug_lsqr = reshape(Afaaug_lsqr, [], 1);

    % Computing r_alpha
    r_alpha = g - Afaaug_lsqr;
    n = length(r_alpha);

    % DP equation
    res_DP = (norm(r_alpha, 1).^2);
    res_reg = (norm(faaug_lsqr, 2).^2);

end
%}



