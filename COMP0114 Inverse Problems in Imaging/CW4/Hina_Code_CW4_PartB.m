close all;
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      COMP0114 - Coursework 4                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                              PART B                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Advanced Topic 1: Inpainting in Sinogram Space            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Starting point: replicating the situation

% Original phantom
f = phantom('Modified Shepp-Logan', 64);

% Original sinogram
angles = [0:1:179];
g = radon(f, angles);
g = g/max(g(:));

% Undersampled sinogram (1 every 3) [0:3:179]
g_under = zeros(size(g, 1), size(g, 2));
for kk=1:180
    if mod(kk-1, 3)==0
        g_under(:, kk) = g(:, kk);
    end
end

% Limited angles (60 missing)
g_limited = g;
g_limited(:, 60:120) = 0;


%% Plotting

figure();
cmap = colormap('gray');
subplot(221);
imagesc(f);
title('Original Phantom image');
subplot(222);
imagesc(g);
title('Original sinogram');
subplot(223);
imagesc(g_under);
title('Undersampled sinogram');
subplot(224);
imagesc(g_limited);
title('Limited angles sinogram');


%% Defining the masks

mask_under = ones(size(g, 1), size(g, 2));
mask_limited = ones(size(g, 1), size(g, 2));

% For the limited number of angles, this is just one window
mask_limited(15:80, 60:120) = 0;

% For the undersampled, multiple windows between the measurements
for kk=1:180
    if mod(kk-1, 3)==0
        mask_under(15:80, kk+1:kk+2) = 0;
    end
end


%% Plotting

figure();
subplot(121);
imagesc(imoverlay(g_under, 1-mask_under));
title('Undersampled sinogram + mask');
subplot(122);
imagesc(imoverlay(g_limited, 1-mask_limited));
title('Undersampled sinogram + mask');


%% Plotting masks + original data

figure();
cmap = colormap('gray');
subplot(321);
imagesc(f);
title('Original Phantom image');
subplot(322);
imagesc(g);
title('Original sinogram');
subplot(323);
imagesc(g_under);
title('Undersampled sinogram');
subplot(324);
imagesc(g_limited);
title('Limited angles sinogram');
subplot(325);
imagesc(imoverlay(g_under, 1-mask_under));
title('Undersampled sinogram + mask');
subplot(326);
imagesc(imoverlay(g_limited, 1-mask_limited));
title('Undersampled sinogram + mask');


%% Obtaining the identity matrix over Omega domain

% Identity matrix
I_Omega = speye(size(g, 1)*size(g, 2));

% Indices where the masks are OK (==1)
mind_under = find(mask_under==1);
mind_limited = find(mask_limited==1);

% Obtaining the identity requested
I_Omega_under = I_Omega(mind_under, :);
I_Omega_limited = I_Omega(mind_limited, :);


%% Reconstruction - parameters & data

% Regularisation parameter
alpha_under = 0.0001; %TODO: DP
alpha_limited = 1; 

% Data
g_data_limited = I_Omega_limited * reshape(g, [], 1);
g_data_under = I_Omega_under * reshape(g, [], 1);


%% Reconstruction - solver

% Solver
f_limited_pcg = pcg(@(x)ATA(x, alpha_limited, I_Omega_limited), I_Omega_limited.' * g_data_limited, 1e-08, 500);
f_under_pcg = pcg(@(x)ATA(x, alpha_under, I_Omega_under), I_Omega_under.' * g_data_under, 1e-08, 500);

% Reshaping (solver output is a vector, we want a matrix/image)
f_limited_pcg = reshape(f_limited_pcg, size(g, 1), size(g, 2));
f_under_pcg = reshape(f_under_pcg, size(g, 1), size(g, 2));


%% Processing

% Copying
f_limited_pcg_p = f_limited_pcg;
f_under_pcg_p = f_under_pcg;

% Removing too little and too high values
f_limited_pcg_p(f_limited_pcg_p<1e-07) = 0;
f_limited_pcg_p(f_limited_pcg_p>3) = 1.2;

f_under_pcg_p(f_under_pcg_p<1e-07) = 0;
f_under_pcg_p(f_under_pcg_p>3) = 1.2;

% Normalizing
f_limited_pcg = f_limited_pcg/max(f_limited_pcg(:));
f_under_pcg = f_under_pcg/max(f_under_pcg(:));
f_limited_pcg_p = f_limited_pcg_p/max(f_limited_pcg_p(:));
f_under_pcg_p = f_under_pcg_p/max(f_under_pcg_p(:));


%% Plotting (sinograms)

figure();
subplot(221);
imagesc(g_under);
title('Undersampled sinogram');
subplot(222);
imagesc(g_limited);
title('Limited angles sinogram');
subplot(223);
imagesc(f_under_pcg);
title('Undersampled reconstructed sinogram');
subplot(224);
imagesc(f_limited_pcg);
title('Limited angles reconstructed sinogram');
sgtitle('Non-processed sinograms');

figure();
subplot(221);
imagesc(iradon(g_under, angles));
title('Undersampled sinogram');
subplot(222);
imagesc(iradon(g_limited, angles));
title('Limited angles sinogram');
subplot(223);
imagesc(iradon(f_under_pcg, angles));
title('Undersampled reconstructed sinogram');
subplot(224);
imagesc(iradon(f_limited_pcg, angles));
title('Limited angles reconstructed sinogram');
sgtitle('Non-processed sinograms');

figure();
subplot(221);
imagesc(g_under);
title('Undersampled sinogram');
subplot(222);
imagesc(g_limited);
title('Limited angles sinogram');
subplot(223);
imagesc(f_under_pcg_p);
title('Undersampled reconstructed sinogram');
subplot(224);
imagesc(f_limited_pcg_p);
title('Limited angles reconstructed sinogram');
sgtitle('Processed sinograms');

figure();
subplot(221);
imagesc(iradon(g_under, angles));
title('Undersampled sinogram');
subplot(222);
imagesc(iradon(g_limited, angles));
title('Limited angles sinogram');
subplot(223);
imagesc(iradon(f_under_pcg_p, angles));
title('Undersampled reconstructed sinogram');
subplot(224);
imagesc(iradon(f_limited_pcg_p, angles));
title('Limited angles reconstructed sinogram');
sgtitle('Processed sinograms');


%% Denoising (data)

% Reshaping
f_limited_pcg_d_data = reshape(iradon(f_limited_pcg, angles), [], 1);
f_under_pcg_d_data = reshape(iradon(f_under_pcg, angles), [], 1);
f_limited_pcg_p_d_data = reshape(iradon(f_limited_pcg_p, angles), [], 1);
f_under_pcg_p_d_data = reshape(iradon(f_under_pcg_p, angles), [], 1);

% Regularisation parameter
alpha_d = 2;


%% Denoising (solver)

f_limited_pcg_d = pcg(@(x)ATAdenoising(x, alpha_d), f_limited_pcg_d_data, 1e-03, 100, [], [], f_limited_pcg_d_data);
f_under_pcg_d = pcg(@(x)ATAdenoising(x, alpha_d), f_under_pcg_d_data, 1e-03, 100, [], [], f_under_pcg_d_data);
f_limited_pcg_p_d = pcg(@(x)ATAdenoising(x, alpha_d), f_limited_pcg_p_d_data, 1e-03, 100, [], [], f_limited_pcg_p_d_data);
f_under_pcg_p_d = pcg(@(x)ATAdenoising(x, alpha_d), f_under_pcg_p_d_data, 1e-03, 100, [], [], f_under_pcg_p_d_data);

f_limited_pcg_d = reshape(f_limited_pcg_d, 66, 66);
f_under_pcg_d = reshape(f_under_pcg_d, 66, 66);
f_limited_pcg_p_d = reshape(f_limited_pcg_p_d, 66, 66);
f_under_pcg_p_d = reshape(f_under_pcg_p_d, 66, 66);


%% Denoising (plotting)

figure();
subplot(241);
imagesc(reshape(f_limited_pcg_p_d_data, 66, 66));
title('Limited noisy image');
subplot(242);
imagesc(f_limited_pcg_p_d);
title('Limited denoised image');
subplot(243);
imagesc(reshape(f_under_pcg_p_d_data, 66, 66));
title('Undersampled noisy image');
subplot(244);
imagesc(f_under_pcg_p_d);
title('Undersampled denoised image');
subplot(245);
%imagesc(f_limited_pcg_p);
imagesc(radon(reshape(f_limited_pcg_p_d_data, 66, 66), angles));
title('Limited noisy sinogram');
subplot(246);
imagesc(radon(f_limited_pcg_p_d, angles));
title('Limited denoised sinogram');
subplot(247);
imagesc(f_under_pcg_p);
title('Undersampled noisy sinogram');
subplot(248);
imagesc(radon(f_under_pcg_p_d, angles));
title('Undersampled denoised sinogram');


%% Denoising (wavelets)

[a_limited, h_limited, v_limited, d_limited] = haart2(reshape(f_limited_pcg_p_d_data, 66, 66));
[a_under, h_under, v_under, d_under] = haart2(reshape(f_under_pcg_p_d_data, 66, 66));

thresh_method = 'h'; %'s' or 'h'
[h_limited_d, v_limited_d, d_limited_d] = thresholdFunction(h_limited, v_limited, d_limited, [1:4], 0.8, thresh_method);
[h_under_d, v_under_d, d_under_d] = thresholdFunction(h_under, v_under, d_under, [1:4], 0.8, thresh_method);

f_limited_p_wave = ihaart2(a_limited, h_limited_d, v_limited_d, d_limited_d);
f_under_p_wave = ihaart2(a_under, h_under_d, v_under_d, d_under_d);


%% Plotting (wavelets)

figure();
subplot(241);
imagesc(reshape(f_limited_pcg_p_d_data, 66, 66));
title('Limited noisy image');
subplot(242);
imagesc(f_limited_p_wave);
title('Limited denoised image');
subplot(243);
imagesc(reshape(f_under_pcg_p_d_data, 66, 66));
title('Undersampled noisy image');
subplot(244);
imagesc(f_under_p_wave);
title('Undersampled denoised image');
subplot(245);
imagesc(f_limited_pcg_p);
title('Limited noisy sinogram');
subplot(246);
imagesc(radon(f_limited_p_wave, angles));
title('Limited denoised sinogram');
subplot(247);
imagesc(f_under_pcg_p);
title('Undersampled noisy sinogram');
subplot(248);
imagesc(radon(f_under_p_wave, angles));
title('Undersampled denoised sinogram');


%% Plotting denoising GENERAL

figure();
subplot(241);
imagesc(f_limited_pcg_p_d);
title('Limited Krylov denoised image');
subplot(242);
imagesc(f_limited_p_wave);
title('Limited Wavelet denoised image');
subplot(243);
imagesc(f_under_pcg_p_d);
title('Undersampled Krylov denoised image');
subplot(244);
imagesc(f_under_p_wave);
title('Undersampled Wavelet denoised image');
subplot(245);
imagesc(radon(f_limited_pcg_p_d, angles));
title('Reconstructed limited Krylov denoised sinogram');
subplot(246);
imagesc(radon(f_limited_p_wave, angles));
title('Reconstructed limited Wavelet denoised sinogram');
subplot(247);
imagesc(radon(f_under_pcg_p_d, angles));
title('Reconstructed undersampled Krylov denoised sinogram');
subplot(248);
imagesc(radon(f_under_p_wave, angles));
title('Reconstructed undersampled Wavelet denoised sinogram');


%% UNDER-SAMPLED
%% GAUSS-JACOBI TV - INIT

img_0 = g_under;
img_k = g_under;  
img_kplusone = zeros(95, 180);
E = bwperim(mask_under);
lambda_reg = 0.01;
a = 0.001;
nb_iter = 1000;


%% GAUSS-JACOBI TV - ITER

for kk=1:nb_iter

for ii=2:95-1
    for jj=2:180-1

        if( (mask_under(ii, jj)==0) || E(ii, jj) ) %Only if we're in DuE

        % Defining the neighborhood
        uO = img_k(ii, jj);
        uN = img_k(ii-1, jj);
        uNE = img_k(ii-1, jj+1);
        uE = img_k(ii, jj+1);
        uSE = img_k(ii+1, jj+1);
        uS = img_k(ii+1, jj);
        uSW = img_k(ii+1, jj-1);
        uW = img_k(ii, jj-1);
        uNW = img_k(ii-1, jj-1);

        
        delta_uE = sqrt((uE-uO).^2 + ((uNE+uN-uS-uSE)/4).^2);
        delta_uS = sqrt((uS-uO).^2 + ((uSE+uE-uW-uSW)/4).^2);
        delta_uW = sqrt((uW-uO).^2 + ((uSW+uS-uN-uNW)/4).^2);
        delta_uN = sqrt((uN-uO).^2 + ((uNW+uW-uE-uNE)/4).^2);
        %{
        delta_uE = sqrt((uE-uO).^2 + ((uNE-uSE)/2).^2);
        delta_uS = sqrt((uS-uO).^2 + ((uSE-uSW)/2).^2);
        delta_uW = sqrt((uW-uO).^2 + ((uSW-uNW)/2).^2);
        delta_uN = sqrt((uN-uO).^2 + ((uNW-uNE)/2).^2);
        %}

        wE = 1/sqrt(a.^2 + delta_uE.^2);
        wS = 1/sqrt(a.^2 + delta_uS.^2);
        wW = 1/sqrt(a.^2 + delta_uW.^2);
        wN = 1/sqrt(a.^2 + delta_uN.^2);

        lambdaO = lambda_reg*E(ii, jj);

        hOO = lambdaO/(wE+wS+wW+wN+lambdaO);

        hOE = wE/(wE+wS+wW+wN+lambdaO);
        hOS = wS/(wE+wS+wW+wN+lambdaO);
        hOW = wW/(wE+wS+wW+wN+lambdaO);
        hON = wN/(wE+wS+wW+wN+lambdaO);

        img_kplusone(ii, jj) = hOE*uE + hOS*uS + hOW*uW + hON*uN + hOO*img_k(ii, jj);

        else

        %img_kplusone(ii, jj) = img_k(ii, jj);

        end
    end
end

%img_k = img_kplusone;
img_k = img_0 + (1-mask_under).*img_kplusone;
img_kplusone = zeros(95, 180);

end

img_k_under = img_k;


%% GAUSS-JACOBI TV - PLOT

figure();
subplot(221);
imagesc(g_under);
subplot(222);
imagesc(img_k);
subplot(223);
imagesc(iradon(g_under, angles));
subplot(224);
imagesc(iradon(img_k, angles));


%% LIMITED ANGLES
%% GAUSS-JACOBI TV - INIT

img_0 = g_limited;
img_k = g_limited;  
img_kplusone = zeros(95, 180);
E = bwperim(mask_limited);
lambda_reg = 1;
a = 0.001;
nb_iter = 1000;


%% GAUSS-JACOBI TV - ITER

for kk=1:nb_iter

for ii=2:95-1
    for jj=2:180-1

        if( (mask_limited(ii, jj)==0) || E(ii, jj) ) %Only if we're in DuE

        % Defining the neighborhood
        uO = img_k(ii, jj);
        uN = img_k(ii-1, jj);
        uNE = img_k(ii-1, jj+1);
        uE = img_k(ii, jj+1);
        uSE = img_k(ii+1, jj+1);
        uS = img_k(ii+1, jj);
        uSW = img_k(ii+1, jj-1);
        uW = img_k(ii, jj-1);
        uNW = img_k(ii-1, jj-1);

        %{
        delta_uE = sqrt((uE-uO).^2 + ((uNE+uN-uS-uSE)/4).^2);
        delta_uS = sqrt((uS-uO).^2 + ((uSE+uE-uW-uSW)/4).^2);
        delta_uW = sqrt((uW-uO).^2 + ((uSW+uS-uN-uNW)/4).^2);
        delta_uN = sqrt((uN-uO).^2 + ((uNW+uW-uE-uNE)/4).^2);
        %}
        delta_uE = sqrt((uE-uO).^2 + ((uNE-uSE)/2).^2);
        delta_uS = sqrt((uS-uO).^2 + ((uSE-uSW)/2).^2);
        delta_uW = sqrt((uW-uO).^2 + ((uSW-uNW)/2).^2);
        delta_uN = sqrt((uN-uO).^2 + ((uNW-uNE)/2).^2);

        wE = 1/sqrt(a.^2 + delta_uE.^2);
        wS = 1/sqrt(a.^2 + delta_uS.^2);
        wW = 1/sqrt(a.^2 + delta_uW.^2);
        wN = 1/sqrt(a.^2 + delta_uN.^2);
        
        if( E(ii, jj) )
        lambdaO = lambda_reg*E(ii, jj);
        else
        lambdaO = 0;
        end

        hOO = lambdaO/(wE+wS+wW+wN+lambdaO);

        hOE = wE/(wE+wS+wW+wN+lambdaO);
        hOS = wS/(wE+wS+wW+wN+lambdaO);
        hOW = wW/(wE+wS+wW+wN+lambdaO);
        hON = wN/(wE+wS+wW+wN+lambdaO);

        img_kplusone(ii, jj) = hOE*uE + hOS*uS + hOW*uW + hON*uN + hOO*img_k(ii, jj);

        else

        img_kplusone(ii, jj) = img_k(ii, jj);

        end

    end
end

%img_k = img_kplusone;
img_k = img_0 + (1-mask_limited).*img_kplusone;
img_kplusone = zeros(95, 180);

end

img_k_limited = img_k;


%% GAUSS-JACOBI TV - PLOT

figure();
subplot(121);
imagesc(g_limited);
subplot(122);
%imagesc(g_limited + (1-mask_limited).*img_k);
imagesc(img_k);



%% GAUSS-JACOBI TV - FINAL PLOT

figure();
subplot(221);
imagesc(g_under);
title('Undersampled angles sinogram');
subplot(222);
imagesc(img_k_under);
title('Undersampled angles TV reconstruction');
subplot(223);
imagesc(g_limited);
title('Limited angles sinogram');
subplot(224);
imagesc(img_k_limited);
title('Limited angles TV reconstruction');


%% Gradient of the original sinogram (used in the Anisotropic section)

[Gm, ~] = imgradient(g, 'sobel');
Gm = abs(Gm);
Gm(Gm<0.2) = 0;
Gm = Gm/max(Gm(:));

figure();
imagesc(Gm);


%% ANISOTROPIC
%% UNDER-SAMPLED
%% GAUSS-JACOBI TV - INIT

img_0 = [zeros(95, 1) g_under zeros(95, 1)];
img_k = [zeros(95, 1) g_under zeros(95, 1)];  
img_kplusone = zeros(95, 182);
E = bwperim(mask_under); 
lambda_reg = 0.1;
a = 0.001;
nb_iter = 50000;
%c1 = 0.001;
c2 = 1;


%% GAUSS-JACOBI TV - ITER

for kk=1:nb_iter

for ii=2:95-1
    for jj=2:180+1

        if( (mask_under(ii, jj-1)==0) || E(ii, jj-1) ) %Only if we're in DuE

        if(Gm(ii, jj-1) ~= 0)
            c1 = 0.001;
        else
            c1 = Gm(ii, jj-1);
        end

        % Defining the neighborhood
        uO = img_k(ii, jj);
        uN = img_k(ii-1, jj);
        uNE = img_k(ii-1, jj+1);
        uE = img_k(ii, jj+1);
        uSE = img_k(ii+1, jj+1);
        uS = img_k(ii+1, jj);
        uSW = img_k(ii+1, jj-1);
        uW = img_k(ii, jj-1);
        uNW = img_k(ii-1, jj-1);

        %{
        delta_uE = sqrt(c1*(uE-uO).^2 + c2*((uNE+uN-uS-uSE)/4).^2);
        delta_uS = sqrt(c1*(uS-uO).^2 + c2*((uSE+uE-uW-uSW)/4).^2);
        delta_uW = sqrt(c1*(uW-uO).^2 + c2*((uSW+uS-uN-uNW)/4).^2);
        delta_uN = sqrt(c1*(uN-uO).^2 + c2*((uNW+uW-uE-uNE)/4).^2);
        %}
        delta_uE = sqrt(c1*(uE-uO).^2 + c2*((uNE-uSE)/2).^2);
        delta_uS = sqrt(c1*(uS-uO).^2 + c2*((uSE-uSW)/2).^2);
        delta_uW = sqrt(c1*(uW-uO).^2 + c2*((uSW-uNW)/2).^2);
        delta_uN = sqrt(c1*(uN-uO).^2 + c2*((uNW-uNE)/2).^2);
        

        wE = 1/sqrt(a.^2 + delta_uE.^2);
        wS = 1/sqrt(a.^2 + delta_uS.^2);
        wW = 1/sqrt(a.^2 + delta_uW.^2);
        wN = 1/sqrt(a.^2 + delta_uN.^2);

        lambdaO = lambda_reg*E(ii, jj-1);

        hOO = lambdaO/(wE+wS+wW+wN+lambdaO);

        hOE = wE/(wE+wS+wW+wN+lambdaO);
        hOS = wS/(wE+wS+wW+wN+lambdaO);
        hOW = wW/(wE+wS+wW+wN+lambdaO);
        hON = wN/(wE+wS+wW+wN+lambdaO);

        img_kplusone(ii, jj) = hOE*uE + hOS*uS + hOW*uW + hON*uN + hOO*img_k(ii, jj);

        else

        %img_kplusone(ii, jj) = img_k(ii, jj);

        end
    end
end

%img_k = img_kplusone;
img_k = img_0 + (1-([zeros(95, 1) mask_under zeros(95, 1)])).*img_kplusone;
img_kplusone = zeros(95, 182);

end


%% GAUSS-JACOBI TV - PLOT

img_k = img_k(:, 2:181);
%img_k_directional_tv_under = img_k;


figure();
subplot(221);
imagesc(g_under);
subplot(222);
imagesc(img_k);
subplot(223);
imagesc(iradon(g_under, angles));
subplot(224);
imagesc(iradon(img_k, angles));


%% LIMITED ANGLES
%% GAUSS-JACOBI TV - INIT

% For the limited number of angles, this is just one window
j_min = 60;
j_max = 120;
mask_limited = ones(95, 180);
mask_limited(15:80, j_min:j_max) = 0;
E = bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-1:j_max+1) = 0;
E = E + bwperim(mask_limited);
% Below is to add more masking area
%{ 
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-2:j_max+2) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-3:j_max+3) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-4:j_max+4) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-5:j_max+5) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-6:j_max+6) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-7:j_max+7) = 0;
E = E + bwperim(mask_limited);
mask_limited = ones(95, 180);
mask_limited(15:80, j_min-8:j_max+8) = 0;
E = E + bwperim(mask_limited);
%}
E(E>1) = 1;


figure();
imagesc(E);

img_0 = g_limited;
img_k = g_limited;  
img_kplusone = zeros(95, 180);
lambda_reg = 0.5;
a = 0.001;
nb_iter = 500000;
c1 = 0.001;
c2 = 1;


%% GAUSS-JACOBI TV - ITER

figure();

for kk=1:nb_iter

for ii=2:95-1
    for jj=2:180-1

        if( (mask_limited(ii, jj)==0) || E(ii, jj) ) %Only if we're in DuE

        if(Gm(ii, jj) ~= 0)
            c1 = 0.001;
        else
            c1 = Gm(ii, jj);
        end

        % Defining the neighborhood
        uO = img_k(ii, jj);
        uN = img_k(ii-1, jj);
        uNE = img_k(ii-1, jj+1);
        uE = img_k(ii, jj+1);
        uSE = img_k(ii+1, jj+1);
        uS = img_k(ii+1, jj);
        uSW = img_k(ii+1, jj-1);
        uW = img_k(ii, jj-1);
        uNW = img_k(ii-1, jj-1);

        
        delta_uE = sqrt(c1*(uE-uO).^2 + c2*((uNE+uN-uS-uSE)/4).^2);
        delta_uS = sqrt(c1*(uS-uO).^2 + c2*((uSE+uE-uW-uSW)/4).^2);
        delta_uW = sqrt(c1*(uW-uO).^2 + c2*((uSW+uS-uN-uNW)/4).^2);
        delta_uN = sqrt(c1*(uN-uO).^2 + c2*((uNW+uW-uE-uNE)/4).^2);
        %{
        delta_uE = sqrt(c1*(uE-uO).^2 + c2*((uNE-uSE)/2).^2);
        delta_uS = sqrt(c1*(uS-uO).^2 + c2*((uSE-uSW)/2).^2);
        delta_uW = sqrt(c1*(uW-uO).^2 + c2*((uSW-uNW)/2).^2);
        delta_uN = sqrt(c1*(uN-uO).^2 + c2*((uNW-uNE)/2).^2);
        %}

        wE = 1/sqrt(a.^2 + delta_uE.^2);
        wS = 1/sqrt(a.^2 + delta_uS.^2);
        wW = 1/sqrt(a.^2 + delta_uW.^2);
        wN = 1/sqrt(a.^2 + delta_uN.^2);
        
        if( E(ii, jj) )
        lambdaO = lambda_reg*E(ii, jj);
        else
        lambdaO = 0;
        end

        hOO = lambdaO/(wE+wS+wW+wN+lambdaO);

        hOE = wE/(wE+wS+wW+wN+lambdaO);
        hOS = wS/(wE+wS+wW+wN+lambdaO);
        hOW = wW/(wE+wS+wW+wN+lambdaO);
        hON = wN/(wE+wS+wW+wN+lambdaO);

        img_kplusone(ii, jj) = hOE*uE + hOS*uS + hOW*uW + hON*uN + hOO*img_k(ii, jj);

        else

        img_kplusone(ii, jj) = img_k(ii, jj);

        end

    end
end

%img_k = img_kplusone;
img_k = img_0 + (1-mask_limited).*img_kplusone;
img_kplusone = zeros(95, 180);

%imagesc(img_k);
%pause(0.00001);

end


%% GAUSS-JACOBI TV - PLOT

%img_k_limited_04 = img_k;
%img_k_limited_00 = img_k;
%img_k_limited_02 = img_k;

figure();
subplot(121);
imagesc(g_limited);
subplot(122);
imagesc( (img_k/max(img_k(:))) + (img_0/max(img_0(:))) );
%imagesc(img_k);


%% ALL TV PLOT

figure();
subplot(231);
imagesc(img_k_under);
title('Undersampled TV reconstruction');
subplot(232);
imagesc(img_k_directional_tv_under);
title('Undersampled directional TV reconstruction');
subplot(233);
imagesc(iradon(img_k_under,angles))
title('FBP of the undersampled TV reconstruction');
subplot(234);
imagesc(img_k_limited);
title('Limited angles TV reconstruction');
subplot(235);
imagesc(img_k_limited_00);
title('Limited angles directional TV reconstruction');
subplot(236);
imagesc(iradon(img_k_limited_00,angles))
title('FBP of the limited angles DTV reconstruction');


%% FINAL PLOT - Denoising

[a_limited, h_limited, v_limited, d_limited] = haart2(iradon(img_k_limited_00,angles));
[a_under, h_under, v_under, d_under] = haart2(iradon(img_k_under,angles));

thresh_method = 'h'; %'s' or 'h'
[h_limited_d, v_limited_d, d_limited_d] = thresholdFunction(h_limited, v_limited, d_limited, [1:4], 0.8, thresh_method);
[h_under_d, v_under_d, d_under_d] = thresholdFunction(h_under, v_under, d_under, [1:4], 0.8, thresh_method);

f_limited_p_wave = ihaart2(a_limited, h_limited_d, v_limited_d, d_limited_d);
f_under_p_wave = ihaart2(a_under, h_under_d, v_under_d, d_under_d);


%% FINAL PLOT - Visualizing

figure();
subplot(231);
imagesc(f);
title('Original phantom');
subplot(232);
imagesc(f_under_p_wave);
title('Undersampled denoised reconstruction');
subplot(233);
imagesc(f_limited_p_wave);
title('Limited angles denoised reconstruction');
subplot(234);
imagesc(g);
title('Original sinogram');
subplot(235);
imagesc(img_k_under);
title('Undersampled TV sinogram');
subplot(236);
imagesc(img_k_limited_00);
title('Limited angles DTV sinogram');


%% Appendix

% Perona-Malik diffusion coefficient

function gamma = perona_malik(f)
    [Df, ~] = imgradient(f, 'prewitt');
    T = 0.2*max(Df(:));
    %T = 0.1*max(Df(:));
    gamma = exp(-Df/T);
end


% Krylov solver for inpainting with Laplacian regularisation

function z = ATA(f, alpha, I_Omega)
    y = I_Omega.' * I_Omega * f;
    z = y - alpha * reshape(del2(reshape(f, 95, 180)), [], 1);
end


% fminunc for TV inpainting
function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ATATVoptim(x, Avox, I_Omega, alpha, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@IVIM, x, h);
    function sumRes = IVIM(x)
        % Parameters
        x_reshaped = reshape(x, 95, 180);
        [Df, ~] = imgradient(x_reshaped, 'prewitt');
        Df = Df/max(Df(:));
        TV = sum(sum(Df));
        x = x + alpha*reshape(Df, [], 1);
        %x = I_Omega.' * I_Omega * x;
        % To minimize
        sumRes = sum((Avox - x).^2) + alpha*TV;
        %disp(sumRes);
    end
end


% Krylov solver for inpainting with TV regularisation

function z = ATATV(f, alpha, I_Omega)
    f_reshaped = reshape(f, 95, 180);
    [Df, ~] = imgradient(f_reshaped, 'prewitt');
    Df = abs(Df);
    T =  0.2*max(Df(:));
    %Df = Df/max(Df(:));
    y = I_Omega.' * I_Omega * f;
    y_reshaped = reshape(y, 95, 180);
    z = y_reshaped - alpha * Df;
    z = reshape(z, [], 1);
end


% Krylov solver for denoising with 0-order regularisation

function z = ATAdenoising(f, alpha)
    f_reshaped = reshape(f, 66, 66);
    gamma = perona_malik(f_reshaped);
    %gamma(gamma<0.2) = 0;
    y = f_reshaped;
    z = y - alpha .* gamma .* del2(f_reshaped);
    %z = y - alpha * del2(f_reshaped);
    z = reshape(z, [], 1);
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


% End