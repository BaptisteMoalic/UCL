close all;
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            Coursework 3                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fixing the random seed
rng(0,'twister');


%% Constructing matrix A

A = rand(300,1024);

U = orth(A);
V = orth(A');
W = eye(300);
%W = diag(ones(300, 1));

A = U * W * V';


%% Looking at the spectrum of A

figure();
plot(diag(W));
title('SVD spectrum of A');


%% Creating the signal

sig = zeros(1024, 1);

% Selecting only n non-zero values
n = 50;
non_zero_idx = randperm(1024, n);

for kk=1:n
    if(unifrnd(0, 1) > 0.5)
        sig(non_zero_idx(kk)) = 1;
    else
        sig(non_zero_idx(kk)) = -1;
    end
end

%plot(sig);


%% Noisy observations

sigma = 0.05;
b = A * sig + sigma * randn(300, 1);

%plot(b);


%% Optimisation

sig_est = l1eq_pd(zeros(1024, 1), A, A, b);

figure();
plot(sig_est);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title({'Recovery of a sparse signal with noisy measurements','\sigma = '+string(sigma)})


%% Formula giving us the number of non-zeros

n2 = 300;
m2 = 1024;
abs_U = abs(U);
mu = sqrt(m2)*max(abs_U(:));
C = 0.5;
S = C*(1/(mu.^2))*(n2/(log(m2).^6));


%% Plotting for the number of non-zeros

% 50 elements
sig = zeros(1024, 1);
n = 50;
non_zero_idx = randperm(1024, n);
for kk=1:n
    if(unifrnd(0, 1) > 0.5)
        sig(non_zero_idx(kk)) = 1;
    else
        sig(non_zero_idx(kk)) = -1;
    end
end

% peutetre mettre le noise à f direct plutôt qua l'observation

sigma1 = 0.05;
sigma2 = 0.1;
sigma3 = 0.2;
sigma4 = 0.5;

b1 = A * sig + sigma1 * randn(300, 1);
b2 = A * sig + sigma2 * randn(300, 1);
b3 = A * sig + sigma3 * randn(300, 1);
b4 = A * sig + sigma4 * randn(300, 1);

sig_est1 = l1eq_pd(zeros(1024, 1), A, A, b1);
sig_est2 = l1eq_pd(zeros(1024, 1), A, A, b2);
sig_est3 = l1eq_pd(zeros(1024, 1), A, A, b3);
sig_est4 = l1eq_pd(zeros(1024, 1), A, A, b4);

figure(1);
subplot(221);
plot(sig_est1);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma1))
subplot(222);
plot(sig_est2);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma2))
subplot(223);
plot(sig_est3);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma3))
subplot(224);
plot(sig_est4);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma4))
sgtitle('S = 50');

% 100 elements
sig = zeros(1024, 1);
n = 100;
non_zero_idx = randperm(1024, n);
for kk=1:n
    if(unifrnd(0, 1) > 0.5)
        sig(non_zero_idx(kk)) = 1;
    else
        sig(non_zero_idx(kk)) = -1;
    end
end

% peutetre mettre le noise à f direct plutôt qua l'observation

b1 = A * sig + sigma1 * randn(300, 1);
b2 = A * sig + sigma2 * randn(300, 1);
b3 = A * sig + sigma3 * randn(300, 1);
b4 = A * sig + sigma4 * randn(300, 1);

sig_est1 = l1eq_pd(zeros(1024, 1), A, A, b1);
sig_est2 = l1eq_pd(zeros(1024, 1), A, A, b2);
sig_est3 = l1eq_pd(zeros(1024, 1), A, A, b3);
sig_est4 = l1eq_pd(zeros(1024, 1), A, A, b4);

figure(2);
subplot(221);
plot(sig_est1);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma1))
subplot(222);
plot(sig_est2);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma2))
subplot(223);
plot(sig_est3);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma3))
subplot(224);
plot(sig_est4);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma4))
sgtitle('S = 100');

% 250 elements
sig = zeros(1024, 1);
n = 250;
non_zero_idx = randperm(1024, n);
for kk=1:n
    if(unifrnd(0, 1) > 0.5)
        sig(non_zero_idx(kk)) = 1;
    else
        sig(non_zero_idx(kk)) = -1;
    end
end

% peutetre mettre le noise à f direct plutôt qua l'observation

b1 = A * sig + sigma1 * randn(300, 1);
b2 = A * sig + sigma2 * randn(300, 1);
b3 = A * sig + sigma3 * randn(300, 1);
b4 = A * sig + sigma4 * randn(300, 1);

sig_est1 = l1eq_pd(zeros(1024, 1), A, A, b1);
sig_est2 = l1eq_pd(zeros(1024, 1), A, A, b2);
sig_est3 = l1eq_pd(zeros(1024, 1), A, A, b3);
sig_est4 = l1eq_pd(zeros(1024, 1), A, A, b4);

figure(3);
subplot(221);
plot(sig_est1);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma1))
subplot(222);
plot(sig_est2);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma2))
subplot(223);
plot(sig_est3);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma3))
subplot(224);
plot(sig_est4);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma4))
sgtitle('S = 250');

% 500 elements
sig = zeros(1024, 1);
n = 500;
non_zero_idx = randperm(1024, n);
for kk=1:n
    if(unifrnd(0, 1) > 0.5)
        sig(non_zero_idx(kk)) = 1;
    else
        sig(non_zero_idx(kk)) = -1;
    end
end

% peutetre mettre le noise à f direct plutôt qua l'observation

b1 = A * sig + sigma1 * randn(300, 1);
b2 = A * sig + sigma2 * randn(300, 1);
b3 = A * sig + sigma3 * randn(300, 1);
b4 = A * sig + sigma4 * randn(300, 1);

sig_est1 = l1eq_pd(zeros(1024, 1), A, A, b1);
sig_est2 = l1eq_pd(zeros(1024, 1), A, A, b2);
sig_est3 = l1eq_pd(zeros(1024, 1), A, A, b3);
sig_est4 = l1eq_pd(zeros(1024, 1), A, A, b4);

figure(4);
subplot(221);
plot(sig_est1);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma1))
subplot(222);
plot(sig_est2);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma2))
subplot(223);
plot(sig_est3);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma3))
subplot(224);
plot(sig_est4);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title('\sigma = '+string(sigma4))
sgtitle('S = 500');


%% Changing the spectrum of A

W = diag(exp(-(1:300)/50));

A = U * W * V';


%% Looking at the new spectrum of A

figure();
plot(diag(W));
title('SVD spectrum of A');

%% Noisy observations

sigma = 0.001;
b = A * sig + sigma * randn(300, 1);

%plot(b);


%% Optimisation

sig_est = l1eq_pd(zeros(1024, 1), A, A, b);

figure();
plot(sig_est);
hold on;
plot(sig, '+');
hold off;
legend('Estimated', 'Ground truth');
title({'Recovery of a sparse signal with noisy measurements','\sigma = '+string(sigma)})
%title('Recovery of a sparse signal without noise')




