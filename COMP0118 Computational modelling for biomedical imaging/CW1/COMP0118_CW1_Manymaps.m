clear;
close all;


%% Importing the data

load('data_p1/data.mat');
qhat = load('data_p1/bvecs');
bvals = 1000 * sum(qhat.*qhat);
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);


% Voxel map of interest
vox72 = dwis(:, :, :, 72);
size_vox72 = size(vox72);
x_i = size_vox72(2);
y_j = size_vox72(3);

nb_of_tries = 4; % Estimated thanks to the above parts

% Desgin matrix for linear estimation
Gdesign = [ones(1, length(bvals)); -bvals].';

G = zeros(108, 7); % We have 108 data points (rows) and 7 unknowns (columns)
G(:, 1) = ones(108, 1);
G(:, 2) = (-bvals .* (qhat(1, :).^2)).';
G(:, 3) = (-2*bvals .* (qhat(1, :).*qhat(2, :))).';
G(:, 4) = (-2*bvals .* (qhat(1, :).*qhat(3, :))).';
G(:, 5) = (-bvals .* (qhat(2, :).^2)).';
G(:, 6) = (-2*bvals .* (qhat(2, :).*qhat(3, :))).';
G(:, 7) = (-bvals .* (qhat(3, :).^2)).';
            
N = 5;
K = length(bvals);

AIC_bs = zeros(x_i, y_j);
AIC_zs = zeros(x_i, y_j);
AIC_zst = zeros(x_i, y_j);


%% Ball and stick

h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
ballandstick = zeros(x_i, y_j, 5);

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            x = pinv(Gdesign)*log(voxOfInterest); 
            S0_lin = max(2500, exp(x(1)));
            d_lin = max(0, x(2));
            startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];

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
            AIC_bs(ii, jj) = 2*N + K*log((1/K)*BallStickSSDUNC(best_params, voxOfInterest, bvals, qhat));
            ballandstick(ii, jj, :) = best_params;
        end
    end
end
toc

%% Zeppelin and stick

h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
zeppelinandstick = zeros(x_i, y_j, 5);

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            x = pinv(Gdesign)*log(voxOfInterest); 
            S0_lin = max(2500, exp(x(1)));
            d_lin = max(0, x(2));
            startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];

            % Computing the x results vector
            x_res = pinv(G) * log(voxOfInterest);
            
            % Recovering the elements of S(0,0) and D from x_res
            D_res = [[x_res(2), x_res(3), x_res(4)]; [x_res(3), x_res(5), x_res(6)]; [x_res(4), x_res(6), x_res(7)]];

            e = eig(D_res);
            lambda1 = max(e);
            lambda2 = min(e);

            % We start by computing a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=ZeppelinStickSSDOptim3(startx_0, voxOfInterest, bvals, qhat, lambda1, lambda2, h);
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
                        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=ZeppelinStickSSDOptim3(startx_0, voxOfInterest, bvals, qhat, lambda1, lambda2, h);
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
            AIC_zs(ii, jj) = 2*N + K*log((1/K)*ZeppelinStickSSDUNC(best_params, voxOfInterest, bvals, qhat, lambda1, lambda2));
            zeppelinandstick(ii, jj, :) = best_params;
        end
    end
end
toc


%% Zeppelin and stick WITH TORTUOSITY

h=optimset('MaxFunEvals',9000,'Algorithm','quasi-newton','TolX',5e-6,'TolFun',5e-6,'display','off');
zeppelinandsticktortuosity = zeros(x_i, y_j, 5);

tic
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            x = pinv(Gdesign)*log(voxOfInterest); 
            S0_lin = max(2500, exp(x(1)));
            d_lin = max(0, x(2));
            startx_0 = [sqrt(S0_lin) sqrt(d_lin) sqrt(-log(5e-01)) 1 1];

            % Computing the x results vector
            x_res = pinv(G) * log(voxOfInterest);
            
            % Recovering the elements of S(0,0) and D from x_res
            D_res = [[x_res(2), x_res(3), x_res(4)]; [x_res(3), x_res(5), x_res(6)]; [x_res(4), x_res(6), x_res(7)]];

            e = eig(D_res);
            lambda1 = max(e);
            lambda2 = min(e);

            % We start by computing a first estimate
            ouaisok = true;
            while(ouaisok)
                try
                   [parameter_hat,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim3(startx_0, voxOfInterest, bvals, qhat, lambda1, h);
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
                        startx = startx_0 + [normrnd(5, 1) normrnd(0, 1) normrnd(0, 0.5) normrnd(0, 1) normrnd(0, 1)];
                        [parameter_hat,RESNORM,~,~]=ZeppelinStickTortuositySSDOptim3(startx_0, voxOfInterest, bvals, qhat, lambda1, h);
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
            AIC_zst(ii, jj) = 2*N + K*log((1/K)*ZeppelinStickTortuositySSDUNC(best_params, voxOfInterest, bvals, qhat, lambda1));
            zeppelinandsticktortuosity(ii, jj, :) = best_params;
        end
    end
end
toc


%% Building the AIC map

AIC_map = zeros(x_i, y_j, 3);
for ii=1:x_i    
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            different_AIC = [AIC_bs(ii, jj) AIC_zs(ii, jj) AIC_zst(ii, jj)];
            [val_idx, idx] = min(different_AIC);
            if(idx==1)
                AIC_map(ii, jj, :) = [1 0 0]; % Ball and stick => RED
            elseif(idx==2)
                AIC_map(ii, jj, :) = [0 1 0]; % Zeppelin and stick => GREEN
            else
                AIC_map(ii, jj, :) = [0 0 1];  % Zeppelin and stick w/ tortuosity => BLUE
            end
        end
    end
end


%% Plotting


figure();
imshow(flipud(pagetranspose(AIC_map)), []);
hold on;
scatter(0, 0, 'red');
hold on;
scatter(0, 0, 'green');
hold on;
scatter(0, 0, 'blue');
hold off;
legend('Ball-and-stick', 'Zeppelin-and-stick', 'Zeppelin-and-stick with tortuosity');
title('Map showing the best AIC criterion for each voxel');


%% Computing the different RESNORMs

RESNORM_bs = zeros(x_i, y_j);
RESNORM_zs = zeros(x_i, y_j);
RESNORM_zst = zeros(x_i, y_j);
RESNORM_AIC_map = zeros(x_i, y_j);
RESNORM_AIC_weighted = zeros(x_i, y_j);
weightedmodel = zeros(x_i, y_j, 5);

for ii=1:x_i
    for jj=1:y_j
        voxOfInterest = vox72(:, ii, jj); % Choosing our voxel
        if(min(voxOfInterest)>0) % Only computing brain-voxels
            RESNORM_bs(ii, jj) = log(BallStickSSDUNC(ballandstick(ii, jj, :), voxOfInterest, bvals, qhat)); % Computing the RESNORM for each map model
            RESNORM_zs(ii, jj) = log(ZeppelinStickSSDUNC(zeppelinandstick(ii, jj, :), voxOfInterest, bvals, qhat, lambda1, lambda2));
            RESNORM_zst(ii, jj) = log(ZeppelinStickTortuositySSDUNC(zeppelinandsticktortuosity(ii, jj, :), voxOfInterest, bvals, qhat, lambda1));
            if(AIC_map(ii, jj, 1)==1)
                RESNORM_AIC_map(ii, jj) = log(BallStickSSDUNC(ballandstick(ii, jj, :), voxOfInterest, bvals, qhat));
            elseif(AIC_map(ii, jj, 2)==1)
                RESNORM_AIC_map(ii, jj) = log(ZeppelinStickSSDUNC(zeppelinandstick(ii, jj, :), voxOfInterest, bvals, qhat, lambda1, lambda2));
            else
                RESNORM_AIC_map(ii, jj) = log(ZeppelinStickTortuositySSDUNC(zeppelinandsticktortuosity(ii, jj, :), voxOfInterest, bvals, qhat, lambda1));
            end
            
            % We will now compute the Akaike's weights to compute an
            % optimal "average" of the estimate
            different_AIC = [AIC_bs(ii, jj) AIC_zs(ii, jj) AIC_zst(ii, jj)];
            [min_AIC, idx] = min(different_AIC);

            DeltaAIC_bs = exp((-1/2)*(AIC_bs(ii, jj) - min_AIC));
            DeltaAIC_zs = exp((-1/2)*(AIC_zs(ii, jj) - min_AIC));
            DeltaAIC_zst = exp((-1/2)*(AIC_zst(ii, jj) - min_AIC));
            
            denom_wAIC = DeltaAIC_bs + DeltaAIC_zs + DeltaAIC_zst;

            wAIC_bs = DeltaAIC_bs/denom_wAIC;
            wAIC_zs = DeltaAIC_zs/denom_wAIC;
            wAIC_zst = DeltaAIC_zst/denom_wAIC;

            weightedmodel(ii, jj, :) = wAIC_bs*ballandstick(ii, jj, :) + wAIC_zs*zeppelinandstick(ii, jj, :) + wAIC_zst*zeppelinandsticktortuosity(ii, jj, :);
            RESNORM_AIC_weighted(ii, jj) = log(wAIC_bs*BallStickSSDUNC(weightedmodel(ii, jj, :), voxOfInterest, bvals, qhat) + wAIC_zs*ZeppelinStickSSDUNC(weightedmodel(ii, jj, :), voxOfInterest, bvals, qhat, lambda1, lambda2) + wAIC_zst*ZeppelinStickTortuositySSDUNC(weightedmodel(ii, jj, :), voxOfInterest, bvals, qhat, lambda1));

        end
    end
end

%% Processing the NaNs/Inf
% _p for processed

RESNORM_bs_p = RESNORM_bs;
RESNORM_bs_p(isnan(RESNORM_bs_p)) = 0;
RESNORM_bs_p(isinf(RESNORM_bs_p)) = 0;
RESNORM_zs_p = RESNORM_zs;
RESNORM_zs_p(isnan(RESNORM_zs_p)) = 0;
RESNORM_zs_p(isinf(RESNORM_zs_p)) = 0;
RESNORM_zst_p = RESNORM_zst;
RESNORM_zst_p(isnan(RESNORM_zst_p)) = 0;
RESNORM_zst_p(isinf(RESNORM_zst_p)) = 0;
RESNORM_AIC_map_p = RESNORM_AIC_map;
RESNORM_AIC_map_p(isnan(RESNORM_AIC_map_p)) = 0;
RESNORM_AIC_map_p(isinf(RESNORM_AIC_map_p)) = 0;
RESNORM_AIC_weighted_p = RESNORM_AIC_weighted;
RESNORM_AIC_weighted_p(isnan(RESNORM_AIC_weighted_p)) = 0;
RESNORM_AIC_weighted_p(isinf(RESNORM_AIC_weighted_p)) = 0;


%% Plotting Akaike's weights

[max_norm, max_norm_idx] = max([max(RESNORM_bs_p(:)) max(RESNORM_bs_p(:)) max(RESNORM_bs_p(:)) max(RESNORM_bs_p(:)) max(RESNORM_bs_p(:))]); 

figure();
subplot(151);
imshow(flipud(pagetranspose(RESNORM_bs_p/max_norm)), []);
title('Ball-and-stick');
str1 = 'Average: '+string(mean(RESNORM_bs_p(:)));
str2 = 'Sum: '+string(sum(RESNORM_bs_p(:)));
xlabel({str1, str2});
subplot(152);
imshow(flipud(pagetranspose(RESNORM_zs_p/max_norm)), []);
title('Zeppelin-and-stick');
str1 = 'Average: '+string(mean(RESNORM_zs_p(:)));
str2 = 'Sum: '+string(sum(RESNORM_zs_p(:)));
xlabel({str1, str2});
subplot(153);
imshow(flipud(pagetranspose(RESNORM_zst_p/max_norm)), []);
title('Zeppelin-and-stick with tortuosity');
str1 = 'Average: '+string(mean(RESNORM_zst_p(:)));
str2 = 'Sum: '+string(sum(RESNORM_zst_p(:)));
xlabel({str1, str2});
subplot(154);
imshow(flipud(pagetranspose(RESNORM_AIC_map_p/max_norm)), []);
title('Best model for each voxel');
str1 = 'Average: '+string(mean(RESNORM_AIC_map_p(:)));
str2 = 'Sum: '+string(sum(RESNORM_AIC_map_p(:)));
xlabel({str1, str2});
subplot(155);
imshow(flipud(pagetranspose(RESNORM_AIC_weighted_p/max_norm)), []);
title('Averaged model with Akaike''s weights');
str1 = 'Average: '+string(mean(RESNORM_AIC_weighted_p(:)));
str2 = 'Sum: '+string(sum(RESNORM_AIC_weighted_p(:)));
xlabel({str1, str2});
sgtitle({'RESNORM for different models', 'averaged by the maximum norm of all error maps'})




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


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickSSDOptim3(x, Avox, bvals, qhat, lambda1, lambda2, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@ZeppelinStickSSD, x, h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickSSDOptim4(x, Avox, bvals, qhat, lambda1, lambda2, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@ZeppelinStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickTortuositySSDOptim3(x, Avox, bvals, qhat, lambda1, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@ZeppelinStickSSD, x, h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = ZeppelinStickTortuositySSDOptim4(x, Avox, bvals, qhat, lambda1, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@ZeppelinStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = ZeppelinStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        % Synthesize the signals according to the model
        fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
        fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
        S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end

function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoBallStickSSDOptim3(x, Avox, bvals, qhat, h)
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x, h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
    end
end


function [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = TwoBallStickSSDOptim4(x, Avox, bvals, qhat, h)
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [0 0 0 -pi/2 -pi 0 -pi/2 -pi];
    ub = [Inf Inf 1 pi/2 pi 1 pi/2 pi];
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fmincon(@BallStickSSD, x, A, b, Aeq, beq, lb, ub, [], h);
    function sumRes = BallStickSSD(x)
        % Extract the parameters
        S0 = x(1);
        diff = x(2);
        f1 = x(3);
        theta1 = x(4);
        phi1 = x(5);
        f2 = x(6);
        theta2 = x(7);
        phi2 = x(8);
        % Synthesize the signals according to the model
        fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
        % Compute the sum of square differences
        sumRes = sum((Avox - S').^2);
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

function sumRes = ZeppelinStickSSD(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickTortuositySSD(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoBallStickSSD(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickSSD(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickTortuositySSD(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1);
    diff = x(2);
    f1 = x(3);
    theta1 = x(4);
    phi1 = x(5);
    f2 = x(6);
    theta2 = x(7);
    phi2 = x(8);
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = BallStickSSDUNC(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickSSDUNC(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = ZeppelinStickTortuositySSDUNC(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f = exp(-(x(3)^2));
        theta = (exp(-(x(4)^2)))*pi;
        phi = (exp(-(x(5)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals.*((1-f)*lambda1 + (-f*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoBallStickSSDUNC(x, Avox, bvals, qhat)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals*diff));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickSSDUNC(x, Avox, bvals, qhat, lambda1, lambda2)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*(lambda2 + (lambda1-lambda2).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

function sumRes = TwoZeppelinStickTortuositySSDUNC(x, Avox, bvals, qhat, lambda1)
    % Extract the parameters
    S0 = x(1)^2;
        diff = x(2)^2;
        f1 = exp(-(x(3)^2));
        theta1 = (exp(-(x(4)^2)))*pi;
        phi1 = (exp(-(x(5)^2)))*2*pi;
        f2 = exp(-(x(6)^2));
        theta2 = (exp(-(x(7)^2)))*pi;
        phi2 = (exp(-(x(8)^2)))*2*pi;
    % Synthesize the signals according to the model
    fibdir1 = [cos(phi1)*sin(theta1) sin(phi1)*sin(theta1) cos(theta1)];
        fibdotgrad1 = sum(qhat.*repmat(fibdir1, [length(qhat) 1])');
        fibdir2 = [cos(phi2)*sin(theta2) sin(phi2)*sin(theta2) cos(theta2)];
        fibdotgrad2 = sum(qhat.*repmat(fibdir2, [length(qhat) 1])');
        fibdotgrad = (fibdotgrad1+fibdotgrad2)/2;
        S = S0*(f1*exp(-bvals*diff.*(fibdotgrad1.^2)) + f2*exp(-bvals*diff.*(fibdotgrad2.^2)) + (1-f1-f2)*exp(-bvals.*((1-f1-f2)*lambda1 + ((-f1-f2)*lambda1).*(fibdotgrad.^2))));
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
end

