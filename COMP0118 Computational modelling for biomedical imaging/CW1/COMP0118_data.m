% Loading the data in a separate file so it does not have to be reloaded at
% every run

load('data_p1/data.mat');
qhat = load('data_p1/bvecs');
bvals = 1000 * sum(qhat.*qhat);
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);