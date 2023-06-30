%%
%addpath( '../../StandardTestImages/');
N = 256;
im = double(imread('Cameraman256.png','PNG'));
%im = double(imread('house.png','PNG'));

figure(1); clf
subplot(1,2,1);imagesc(im); colormap(gray);title('original image');

I256 = speye(256*256);

mask = ones(256);
mask(65:200,60:200) = 0;
mask(140:155,175:200) = 0;
mask(210:240,155:195) = 0;
mim = mask.*im;
subplot(1,2,2);imagesc(mim); colormap(gray);title('masked image');

%%
mind = find(mask==1);
A = I256(mind,:);

g = A*reshape(im,[],1); % this is the reduced size data;
%IA = A'*inv(A*A');
%fr = IA*g;
%fim = reshape(fr,256,256);
%
%subplot(2,2,3);imagesc(fim); colormap(gray);
alpha0 = 1;
fra = inv(A'*A + alpha0*I256)*A'*g;
fraim = reshape(fra,256,256);
figure(1);clf;
subplot(2,2,1);imagesc(im); colormap(gray);title('original image');
subplot(2,2,2);imagesc(mim); colormap(gray);title('masked image');
subplot(2,2,3);imagesc(fraim); colormap(gray);title(['recon with 0-Tikononv \alpha= ',num2str(alpha0)]);


%% construct a better prior...
h = 1/(N+1);
D1 = sparse(N,N);
D1(1,1) = 1;
for k = 2:N
    D1(k-1,k) = -1;
    D1(k,k) = 1;
end

Dx = kron(speye(N),D1);
Dy = kron(D1,speye(N));
Lapl = -(Dx'*Dx + Dy'*Dy);

imx = reshape(Dx*reshape(im,[],1),N,N);
imy = reshape(Dy*reshape(im,[],1),N,N);
imlap = reshape(Lapl*reshape(im,[],1),N,N);
figure(2); clf
subplot(2,2,1);imagesc(im); colormap(gray);title('Original f');
subplot(2,2,2);imagesc(imx); colormap(gray);title('f_x');
subplot(2,2,3);imagesc(imy); colormap(gray);title('f_y');
subplot(2,2,4);imagesc(imlap); colormap(gray);title('\nabla^2 f');


alphaIso = 1e-3;
%fra = inv(A'*A + alpha*Lapl)*A'*g; % don't do this !
Atot = [A;sqrt(alphaIso)*Dx; sqrt(alphaIso)*Dy];
gtot = [g;zeros(N*N,1);zeros(N*N,1)];
tic;
fra = Atot\gtot; % this works, but it's quite slow
%fra = lsqr(Atot,gtot); 
%fra = pcg(@(x) JTJH(x, A, -Lapl, alphaIso),A'*g,1e-6,100);
%fra = lsqr(@(x) JTJH(x, A, Lapl, alpha),A'*g,1e-6,100);
toc;

fraim = reshape(fra,N,N);
figure(1);subplot(2,2,4);imagesc(fraim); colormap(gray);title(['recon with 1-Tikononv \alpha= ',num2str(alphaIso)]);


%% repeat with edge weighted Laplacian
alphaAL = 1e1;
gsq = imx.^2 + imy.^2;
gmax = max(max(gsq));
T = gmax/100;
K = spdiags(exp(-reshape(gsq,[],1)/T),0,256*256,256*256);
figure(2);
subplot(2,2,2);imagesc(sqrt(gsq));colormap(gray);title('image gradient');
subplot(2,2,3);imagesc(reshape(diag(K),256,256));colormap(gray);title('\kappa');
DKD = -(Dx'*K*Dx + Dy'*K*Dy);
tic;
fra2 = pcg(@(x) JTJH(x, A, -DKD, alphaAL),A'*g,1e-6,100);
toc;

fra2im = reshape(fra2,256,256);
figure(1);subplot(2,2,3);imagesc(fraim); colormap(gray);title(['recon with 1-Tikononv \alpha= ',num2str(alphaIso)]);
figure(1);subplot(2,2,4);imagesc(fra2im); colormap(gray);title(['recon with edge-weighted Laplacian \alpha= ',num2str(alphaAL)]);

%% now we try PDE with Dirichlet B.c.s

%mask = ones(N,N); mask(floor(0.6*N):ceil(0.7*N),floor(0.6*N):ceil(0.7*N) )=0;
ind2 = find(mask==1);
outd = setdiff([1:N*N],ind2);
g = zeros(N*N,1);
g(ind2) = im(ind2); % this is an "arbitrary" data vector;


outLapl = Lapl;
%outLapl = -testLapl;
outLapl(ind2,:) = 0; % zeros(length(ind2),N*N);

% First try solving the PDE Lap f = g
finP = speye(N*N);
finP(outd,outd) = 0;
f1 = (finP+outLapl)\g;

% next try as a regularisation
alphaInP = alphaIso;
H = (outLapl- alphaInP*h^2*speye(N*N));
H(ind2,ind2) = speye(length(ind2));
f2 = H\g;

figure(6); clf; hold on;
subplot(2,2,1); imagesc(im);colormap(gray)
%subplot(2,2,2); imagesc(mask);colormap(gray)
subplot(2,2,2); imagesc(reshape(g,N,N));colormap(gray)
subplot(2,2,3); imagesc(fraim); colormap(gray);title(['recon with 1-Tikononv \alpha= ',num2str(alphaIso)]);
subplot(2,2,4); imagesc(reshape(f2,N,N));colormap(gray); title(['PDE + Dirichlet b.c.s \alpha= ',num2str(alphaInP)]);

%%

testest = zeros(256, 256);
testest(outd) = im(outd);
figure();
imagesc(testest);


%%

%ouaistest = sparse(del2(speye(256*256)));

testLapl = sparse(256*256, 256*256);
testLapl(1,1) = 2;
testLapl(2,1) = -1;
testLapl(257,1) = -1;
for kk=2:256
    testLapl(kk, kk) = 3;
    testLapl(kk-1, kk) = -1;
    testLapl(kk+1, kk) = -1;
    testLapl(kk+256, kk) = -1;
end
for kk=257:256*256-256-1
    testLapl(kk, kk) = 4;
    testLapl(kk-1, kk) = -1;
    testLapl(kk+1, kk) = -1;
    testLapl(kk+256, kk) = -1;
    testLapl(kk-256, kk) = -1;
end
for kk=256*256-256:256*256-1
    testLapl(kk, kk) = 3;
    testLapl(kk-1, kk) = -1;
    testLapl(kk+1, kk) = -1;
    testLapl(kk-256, kk) = -1;
end
testLapl(end-1, end) = -1;
testLapl(end, end) = 2;
testLapl(end-256, end) = -1;

%%


figure();
subplot(121);
imagesc(del2(im));
subplot(122);
imagesc(testest);











