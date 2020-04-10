% function SimulateMotionArtifacts()
clear all
%% load data
load('Processed_data/simulated_HbO.mat','simulated_HbO')
load('Processed_data/simulated_HbR.mat','simulated_HbR')
load('Processed_data/list.mat')
load('Processed_data/Real_HbO.mat')
load('Processed_data/Real_HbR.mat')

%%
fs = 7.8125;
pt = 512/fs;
t = 1:fs*pt;
seed = 101;
rng(seed);

Resting_matrix = [simulated_HbO;simulated_HbR];

l_HbO = length(simulated_HbO);
l_HbR = length(simulated_HbR);
if l_HbO ~= l_HbR
    fprintf('HbO number is different from HbR\n');
end
n_dist_HbO = n_MA_HbO_list(randi(length(n_MA_HbO_list),l_HbO,1));
n_dist_HbR = n_MA_HbR_list(randi(length(n_MA_HbR_list),l_HbR,1));
n_list = [n_dist_HbO,n_dist_HbR];
n_list = n_list';
peak_n_list = zeros(size(n_list));
shift_n_list = zeros(size(n_list));
% peak parameters
peak_list = [diff_HbO_list(randi(length(diff_HbO_list),l_HbO,1)),...
    diff_HbR_list(randi(length(diff_HbR_list),l_HbR,1))]';
t0_peak_list = fs*pt*rand(l_HbO+l_HbR,1);
b_low = 0;b_high = 1.5;
b_list = b_low + rand(l_HbO+l_HbR,1).*(b_high-b_low);

% shift parameters
shift_list = [diff_HbO_list(randi(length(diff_HbO_list),l_HbO,1)),...
    diff_HbR_list(randi(length(diff_HbR_list),l_HbR,1))]';
trans_list = round(0.25+(1.5-0.25)*rand(l_HbO+l_HbR,1));%shift transition time is from 0.25s to 1.5s
t0_shift_list = round(1+(fs*pt-trans_list*fs-1-1)*rand);%from 1 to fs*pt-transition*25-1

% HRF parameters
amp_HRF_list = [HRF_HbO_list(randi(length(HRF_HbO_list),l_HbO,1)),...
    HRF_HbR_list(randi(length(HRF_HbR_list),l_HbR,1))]';
drift_Hb_list = [Start_HRF_HbO_list(randi(length(Start_HRF_HbO_list),l_HbO,1)),...
    Start_HRF_HbR_list(randi(length(Start_HRF_HbR_list),l_HbR,1))]';
% noise parameters
neg_sign_values = [-1 1];
neg_sign = neg_sign_values(randi(2,l_HbO+l_HbR,1));

noised_HRF_matrix = zeros(size(Resting_matrix));
noise_profile = zeros(size(Resting_matrix));
Resting_profile = zeros(size(Resting_matrix));
HRF_profile = zeros(size(Resting_matrix));

for i = 1:length(Resting_matrix)
    %HRF
    Resting = Resting_matrix(i,:);
    N = n_list(i);
    noise = zeros(size(Resting));
    if N~=0
        for j = 1:N
            r = randi([0 1],1,1);
            if r == 0
                t0 = t0_peak_list(i);
                A = peak_list(i);
                b = b_list(i);
                noise = noise + neg_sign(i) *A.*exp(-abs(t-t0)./(b*fs));
                peak_n_list(i) = peak_n_list(i) + 1;
            else
                transition = trans_list(i);
                start_point = t0_shift_list(i);
                end_point = start_point+transition*fs;
                DC_shift = shift_list(i);
                shift_sim = zeros(fs*pt,1);
                shift_sim(start_point:end_point) = linspace(0,DC_shift,transition*fs+1);
                shift_sim(end_point:end) = DC_shift;
                noise = noise + neg_sign(i)*shift_sim';
                shift_n_list(i) = shift_n_list(i) + 1;
            end
        end
    end
    amp_Hb = amp_HRF_list(i);
    if i <= l_HbO
        HRF = amp_Hb./gamma(t/15/fs);
    else
        HRF = -amp_Hb./gamma(t/15/fs);
    end
    drift_Hb = drift_Hb_list(i);
    HRF = HRF + drift_Hb;
    noised_HRF = Resting + HRF + noise;
    noised_HRF_matrix(i,:) = noised_HRF;
    noise_profile(i,:) = noise;
    Resting_profile(i,:) = Resting;
    HRF_profile(i,:) = HRF;
end

%% find outliers
[m,n] = size(noised_HRF_matrix);
noised_HbO = noised_HRF_matrix(1:m/2,:);
noised_HbR = noised_HRF_matrix(m/2+1:end,:);
noised_HbO_sum = sum(noised_HbO,2);
noised_HbR_sum = sum(noised_HbR,2);
Real_HbO_sum = sum(Real_HbO,2);
Real_HbR_sum = sum(Real_HbR,2);
thres_HbO = max(abs(Real_HbO_sum));
thres_HbR = max(abs(Real_HbR_sum));

index_HbO = find(abs(noised_HbO_sum)>thres_HbO);
index_HbO_std = find(std(noised_HbO,0,2)>0.15*1e-5);
index_HbO = [index_HbO;index_HbO_std];
index_HbR = find(abs(noised_HbR_sum)>thres_HbR);
index_HbR_std = find(std(noised_HbR,0,2)>0.15*1e-5);
index_HbR = [index_HbR;index_HbR_std];
size(index_HbO)
size(index_HbR)
%%
outrage_index = [index_HbO;index_HbR;index_HbO+m/2;index_HbR+m/2];
noised_HRF_matrix(outrage_index,:) = [];
noise_profile(outrage_index,:) = [];
Resting_profile(outrage_index,:) = [];
HRF_profile(outrage_index,:) = [];
% n_list
n_list(outrage_index) = [];
peak_n_list(outrage_index) = [];
shift_n_list(outrage_index) = [];
% peak parameters
peak_list(outrage_index) = [];
t0_peak_list(outrage_index) = [];
b_list(outrage_index) = [];

% shift parameters
shift_list(outrage_index) = [];
trans_list(outrage_index) = [];
t0_shift_list(outrage_index) = [];

% HRF parameters
amp_HRF_list(outrage_index) = [];
drift_Hb_list(outrage_index) = [];
%% plot
HRF_sum = sum(noised_HRF_matrix,2);
figure
plot(HRF_sum)
%% train, val, test 8:1:1
m = size(HRF_profile,1);
p = randperm(m/2);
n_train = round((m/2)*0.8);
n_val = round((m/2)*0.1);
train_idx = [p(1:n_train),p(1:n_train)+m/2];
val_idx = [p(n_train+1:n_train+n_val),p(n_train+1:n_train+n_val)+m/2];
test_idx = [p(n_train+n_val+1:end),p(n_train+n_val+1:end)+m/2];

HRF_train = HRF_profile(train_idx,:);
HRF_val = HRF_profile(val_idx,:);
HRF_test = HRF_profile(test_idx,:);

HRF_train_noised = noised_HRF_matrix(train_idx,:);
HRF_val_noised = noised_HRF_matrix(val_idx,:);
HRF_test_noised = noised_HRF_matrix(test_idx,:);

Noise_train = noise_profile(train_idx,:);
Noise_val = noise_profile(val_idx,:);
Noise_test = noise_profile(test_idx,:);

Resting_train = Resting_profile(train_idx,:);
Resting_val = Resting_profile(val_idx,:);
Resting_test = Resting_profile(test_idx,:);

%% save
save('Processed_data/random_profile.mat','p')
%output
save('Processed_data/HRF_train.mat','HRF_train')
save('Processed_data/HRF_val.mat','HRF_val')
save('Processed_data/HRF_test.mat','HRF_test')
%input
save('Processed_data/HRF_train_noised.mat','HRF_train_noised')
save('Processed_data/HRF_val_noised.mat','HRF_val_noised')
save('Processed_data/HRF_test_noised.mat','HRF_test_noised')
%noise
save('Processed_data/Noise_train.mat','Noise_train')
save('Processed_data/Noise_val.mat','Noise_val')
save('Processed_data/Noise_test.mat','Noise_test')
% Resting
save('Processed_data/Resting_train.mat','Resting_train')
save('Processed_data/Resting_val.mat','Resting_val')
save('Processed_data/Resting_test.mat','Resting_test')
%% noise profile
% number of noise
save('Processed_data/peak_n_list.mat','peak_n_list')
save('Processed_data/shift_n_list.mat','shift_n_list')
% peak parameters
save('Processed_data/peak_list.mat','peak_list')
save('Processed_data/t0_peak_list.mat','t0_peak_list')
save('Processed_data/b_list.mat','b_list')
% shift parameters
save('Processed_data/shift_list.mat','shift_list')
save('Processed_data/trans_list.mat','trans_list')
save('Processed_data/t0_shift_list.mat','t0_shift_list')
% HRF parameters
save('Processed_data/amp_HRF_list.mat','amp_HRF_list')
save('Processed_data/drift_Hb_list.mat','drift_Hb_list')
% noise parameters
save('Processed_data/neg_sign.mat','neg_sign')
%%
figure
hold on
plot(HRF_test(1,:))
plot(HRF_test_noised(1,:))
plot(Noise_test(1,:))