close all;
clear;

%%%%%%%%%%%%%%%%
%    TASK 3    %
%%%%%%%%%%%%%%%%

%% a) Defining the function

% Generating an array of x values
n = 200;
x = linspace(-1, 1, n);

% Generating the steps (i.e defining the function)
y = zeros(1, n);
for kk=1:n
    res_kk = 0;
    if((-0.95 < x(kk)) & (-0.6 >= x(kk)))
        res_kk = res_kk + 1;
    end
    if((-0.6 < x(kk)) & (-0.2 >= x(kk)))
        res_kk = res_kk + 0.2;
    end
    if((-0.2 < x(kk)) & (0.2 >= x(kk)))
        res_kk = res_kk - 0.5;
    end
    if((0.4 < x(kk)) & (0.6 >= x(kk)))
        res_kk = res_kk + 0.7;
    end
    if((0.6 < x(kk)) & (1 >= x(kk)))
        res_kk = res_kk - 0.7;
    end
    y(kk) = res_kk;
end


%% a) Visualising 

figure();
plot(x,y);
ylim([-1 1.1]);

saveas(gcf, 'task3_a', 'png'); %Saving the figure


%% b) Computing A

% We're using our code from Week 2
% G is defined below
A05 = zeros(n); 
A1 = zeros(n); 
A2 = zeros(n); 
delta_n = abs(x(2)-x(1));
for ii=1:n
    for jj=1:n
        A05(ii, jj) = G(x(ii)-x(jj), 0, 0.05, delta_n);
        A1(ii, jj) = G(x(ii)-x(jj), 0, 0.1, delta_n);
        A2(ii, jj) = G(x(ii)-x(jj), 0, 0.2, delta_n);
    end
end

%% b) Visualising

% Scaling A and instanciating the colormap according to the Hints
colorMap = parula(256);
A05img = ceil(A05/max(A05(:))*256);
A1img = ceil(A1/max(A1(:))*256);
A2img = ceil(A2/max(A2(:))*256);
figure();
subplot(1, 3, 1);
imshow(A05img, colorMap);
title("sigma = 0.05")
subplot(1, 3, 2);
imshow(A1img, colorMap);
title("sigma = 0.1")
subplot(1, 3, 3);
imshow(A2img, colorMap);
title("sigma = 0.2")


%% c) Computing the singular values

[~, W05, ~] = svd(A05);
[~, W1, ~] = svd(A1);
[~, W2, ~] = svd(A2);

W05 = spdiags(W05);
W1 = spdiags(W1);
W2 = spdiags(W2);


%% c) Visualizing

figure();
subplot(1, 3, 1);
plot(W05)
title("sigma = 0.05")
subplot(1, 3, 2);
plot(W1)
title("sigma = 0.1")
subplot(1, 3, 3);
plot(W2)
title("sigma = 0.2")
sgtitle("Singular values of the different A matrices according to their variances");


saveas(gcf, 'task3_c', 'png'); %Saving the figure


%% c) Determining the variance of each data

W05var = [W05;W05];
W1var = [W1;W1];
W2var = [W2;W2];

var05 = sum((W05var-mean(W05var)).^2)/n;
var1 = sum((W1var-mean(W1var)).^2)/n;
var2 = sum((W2var-mean(W2var)).^2)/n;

sigma05 = sqrt(var05);
sigma1 = sqrt(var1);
sigma2 = sqrt(var2);


%% c) Fitting a Gaussian to the data

%test = normpdf(x,0,sigma05);
%test = G(x, 0, sigma05, delta_n)
%plot(test);
%hold on;
%plot(W05);
%hold off;


%% d) Convolution of f by A

conv05 = y * A05;
conv1 = y * A1;
conv2 = y * A2;


%% d) Visualizing

figure();
subplot(1, 3, 1);
plot(x,conv05);
title("sigma = 0.5");
subplot(1, 3, 2);
plot(x,conv1);
title("sigma = 1");
subplot(1, 3, 3);
plot(x,conv2);
title("sigma = 2");


%% e) 

% Defining 1D Gaussians for the Fourier Transform
gauss_1d_05 = normpdf(x,0,0.05);
gauss_1d_1 = normpdf(x,0,0.1);
gauss_1d_2 = normpdf(x,0,0.2);

% Fourier Transforms
FT_gauss_1d_05 = fftshift(fft(fftshift(gauss_1d_05)));
FT_gauss_1d_1 = fftshift(fft(fftshift(gauss_1d_1)));
FT_gauss_1d_2 = fftshift(fft(fftshift(gauss_1d_2)));
FT_y = fftshift(fft(fftshift(y)));

% Multiplication term-by-term
FT_y_conv05 = FT_y .* FT_gauss_1d_05;
FT_y_conv1 = FT_y .* FT_gauss_1d_1;
FT_y_conv2 = FT_y .* FT_gauss_1d_2;

% Going back (IFFT)
y_conv05 = fftshift(ifft(fftshift(FT_y_conv05)));
y_conv1 = fftshift(ifft(fftshift(FT_y_conv1)));
y_conv2 = fftshift(ifft(fftshift(FT_y_conv2)));


%% e) Visualising

figure();
subplot(2, 3, 1);
plot(x,conv05);
title("sigma = 0.5");
ylabel("Convolution of the kernel with the matrix A");
subplot(2, 3, 2);
plot(x,conv1);
title("sigma = 1");
subplot(2, 3, 3);
plot(x,conv2);
title("sigma = 2");
sgtitle("test1");
subplot(2, 3, 4);
plot(x,y_conv05);
ylabel("Multiplication by the kernel in the Fourier domain");
subplot(2, 3, 5);
plot(x,y_conv1);
subplot(2, 3, 6);
plot(x,y_conv2);
sgtitle("Filtering a function by a Gaussian kernel");


%% f) Constructing the matrices A with correct boundaries 

newA05 = zeros(2*n, n);
newA05(1:n, :) = A05;
newA1 = zeros(2*n, n);
newA1(1:n, :) = A1;
newA2 = zeros(2*n, n);
newA2(1:n, :) = A2;

for ii=1:n
    newA05(n+1:2*n,ii) = flipud(A05(:,ii));
    newA1(n+1:2*n,ii) = flipud(A1(:,ii));
    newA2(n+1:2*n,ii) = flipud(A2(:,ii));
end

enfinA05 = zeros(n);
enfinA1 = zeros(n);
enfinA2 = zeros(n);

for jj=1:n
    enfinA05(jj:n, jj) = newA05(n+1:2*n+1-jj);
    enfinA1(jj:n, jj) = newA1(n+1:2*n+1-jj);
    enfinA2(jj:n, jj) = newA2(n+1:2*n+1-jj);
end

symA05 = enfinA05.' + A05 + enfinA05;
symA1 = enfinA1.' + A1 + enfinA1;
symA2 = enfinA2.' + A2 + enfinA2;


%% f) Convolution of f by our new matrices

symconv05 = y * symA05;
symconv1 = y * symA1;
symconv2 = y * symA2;


%% f) Visualising and comparing

figure();
subplot(3, 3, 1);
plot(x,conv05);
title("sigma = 0.5");
subplot(3, 3, 2);
plot(x,conv1);
title("sigma = 1");
xlabel("Convolution of the kernel with the matrix A");
subplot(3, 3, 3);
plot(x,conv2);
title("sigma = 2");
sgtitle("test1");
subplot(3, 3, 4);
plot(x,y_conv05);
subplot(3, 3, 5);
plot(x,y_conv1);
xlabel("Multiplication by the kernel in the Fourier domain");
subplot(3, 3, 6);
plot(x,y_conv2);
subplot(3, 3, 7);
plot(x, symconv05);
subplot(3, 3, 8);
plot(x, symconv1);
xlabel("Convolution of the kernel with the symmetrised matrix A");
subplot(3, 3, 9);
plot(x, symconv2);
sgtitle("Filtering a function by a Gaussian kernel");

saveas(gcf, 'task3_f', 'png'); %Saving the figure


%% Appendix - A matrices

figure();
subplot(1, 2, 1);
imshow(A2img, colorMap);
title("Gaussian kernel");
subplot(1, 2, 2);
imshow(symA2/max(symA2(:))*256, colorMap);
title("Gaussian kernel with symmetry");
sgtitle("Two different Gaussian kernels, for sigma=2");

%saveas(gcf, 'task3_kernel', 'png'); %Saving the figure


%% Appendix - SVD of the symmetrised matrix

[~, Wsym2, ~] = svd(symA2);
Wsym2 = spdiags(Wsym2);
figure()
plot(Wsym2);




%% Appendix - functions

function gaussian_val = G(x, mu, sigma, delta_n)
    gaussian_val = (delta_n/(sqrt(2*pi)*sigma)) * exp(-((x-mu).^2)/(2*(sigma.^2)));
end








