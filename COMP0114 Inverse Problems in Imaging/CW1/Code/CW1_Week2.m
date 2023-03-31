close all;
clear;

%%%%%%%%%%%%%%%%
%    TASK 2    %
%%%%%%%%%%%%%%%%

%% 1a

% Generating an array of x values
n = 20;
x = linspace(-1, 1, n);

%% 1b

% We have defined the function G below
mu = 0;
sigma = 0.2;
delta_n = abs(x(2)-x(1));
Gx = G(x, mu, sigma, delta_n);

%% 1c

% Computing A by iterating over two dimensions
A = zeros(length(x), length(x)); 
for ii=1:length(x)
    for jj=1:length(x)
        A(ii, jj) = G(x(ii)-x(jj), mu, sigma, delta_n);
    end
end

%% 1d

% Scaling A and instanciating the colormap according to the Hints
Aimg = ceil(A/max(A(:))*256);
colorMap = parula(256);
imshow(Aimg, colorMap);

saveas(gcf, 'task2_d', 'png'); %Saving the figure

%% 1e

% Computing the SVD
[U, W, V] = svd(A);

% Let's verify that we achieve a correct reconstruction
A_recon = U * W * V.';
norm_recon = norm(A_recon-A);
disp("Norm difference between the original and reconstructed matrices: "+string(norm_recon));

%% 1f

% Storing W as a sparse matrix
W_sparse = spdiags(W);

% Inverting the matrix
W_dagger_sparse = zeros(length(W_sparse), 1);
for ii=1:length(W_sparse)
    if(W_sparse(ii)==0)
        W_dagger_sparse(ii) = 0;
    else
        W_dagger_sparse(ii) = 1/W_sparse(ii);
    end
end

W_dagger = diag(W_dagger_sparse);
spy(W_dagger);

% Verification of the unitary nature of W
norm_wdagw = norm(eye(n)-(W_dagger*W));
norm_wwdag = norm(eye(n)-(W*W_dagger));
disp("n = "+string(n));
disp("Norm difference between Id and WdW: "+string(norm_wdagw));
disp("Norm difference between Id and WWd: "+string(norm_wwdag));
disp(" ");

% Verification of the unitary nature of A
A_dagger = V * W_dagger * U.';
norm_adaga = norm(eye(n)-(A_dagger*A));
norm_aadag = norm(eye(n)-(A*A_dagger));
disp("Norm difference between Id and AdA: "+string(norm_adaga));
disp("Norm difference between Id and AAd: "+string(norm_aadag));
disp(" ");

% Verification of the correct reconstruction of Ad
norm_adagadag = norm(pinv(A)-A_dagger);
disp("Norm difference between Ad computed by MATLAB's pinv and by SVD: "+string(norm_adagadag));


%% 1g

figure(1)
for jj=1:9
    plot(V(:,jj));
    hold on;
end
hold off;
title("Values taken by the 9 first columns of V");
saveas(gcf, 'task2_g9first20', 'png'); %Saving the figure

figure(2)
for jj=n-9:n
    plot(V(:,jj));
    hold on;
end
hold off;
title("Values taken by the 9 last columns of V");
saveas(gcf, 'task2_g9last20', 'png'); %Saving the figure

figure(3);
semilogy(W_sparse);
title("Singular values of A (log-scaled)");
saveas(gcf, 'task2_gsv20', 'png'); %Saving the figure


%% Appendix - functions

function W_dagger = pinvDiag(W)
    n = size(W);
    n = n(1); % We assume that W is diagonal
    W_dagger = zeros(n, n);

    % Iterating
    for ii=1:n
        if(W(ii, ii) == 0)
            W_dagger(ii, ii) = 0; % Avoiding the division by 0
        else
            W_dagger(ii, ii) = 1/W(ii, ii);
        end
    end

end

function gaussian_val = G(x, mu, sigma, delta_n)
    gaussian_val = (delta_n/(sqrt(2*pi)*sigma)) * exp(-((x-mu).^2)/(2*(sigma.^2)));
end