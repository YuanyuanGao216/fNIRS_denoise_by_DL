clear all
close all

%% load data
load('Processed_data/sim_Resting.mat','simulated_HbO','simulated_HbR', 'loc_example')
load('Processed_data/list.mat','diff_HbO_list','diff_HbR_list','d_MA_list');
% make samples equal in HbO and HbR
n_HbO = size(simulated_HbO,1);
n_HbR = size(simulated_HbR,1);
n_sample = min(n_HbO,n_HbR);
simulated_HbO = simulated_HbO(1:n_sample,:);
simulated_HbR = simulated_HbR(1:n_sample,:);
%% define constants
define_constants

%% define n_list: a list of No. of MA for simulated samples 
tp = size(simulated_HbO,2);
t = (1:tp)/fs_new;
n_MA_list = round(d_MA_list * tp);
n_list = n_MA_list(randi(length(n_MA_list),n_sample,1));
n_list = n_list';
% they are either peak or shift
peak_n_list = zeros(size(n_list));
shift_n_list = zeros(size(n_list));
for i = 1:length(n_list)
    peak_n_list(i) = randi(n_list(i));
    shift_n_list(i) = n_list(i) - peak_n_list(i);
end

%%
s = zeros(1,tp);
s((rt):512:tp) = 1;

%% HbO
noised_HRF_HbO_matrix = zeros(size(simulated_HbO));
noise_HbO_matrix = zeros(size(simulated_HbO));
HRF_profile_HbO_matrix = zeros(size(simulated_HbO));

noised_HRF_HbR_matrix = zeros(size(simulated_HbR));
noise_HbR_matrix = zeros(size(simulated_HbR));
HRF_profile_HbR_matrix = zeros(size(simulated_HbR));

for i = 1:n_sample
    Resting_HbO = simulated_HbO(i,:);
    Resting_HbR = simulated_HbR(i,:);
    noise_HbO = zeros(size(Resting_HbO));
    noise_HbR = zeros(size(Resting_HbR));
    N = n_list(i); % number of MA
    n_peak = peak_n_list(i);
    n_shift = shift_n_list(i);
    
    %% add peaks
    for j = 1:n_peak
        t0 = randi(tp); % MA could happen at any time
        noise_HbO = add_peak(noise_HbO,diff_HbO_list,t0);
        noise_HbR = add_peak(noise_HbR,diff_HbR_list,t0);
    end
    
    %% add shifts
    for j = 1:n_shift
        transition = round((0.25 + (1.5 - 0.25)*rand)*fs_new);
        t0 = randi(tp - transition - 1);
        noise_HbO = add_shift(noise_HbO,diff_HbO_list,t0,transition);
        noise_HbR = add_shift(noise_HbR,diff_HbR_list,t0,transition);
    end
    
    %% HRFs
    % the amplitude of HRF is 0.04uMol (0.01-0.1) in Barker 2013, 2 uM in cooper 2012
    % 0.04uMol is too small, here I temporily set amp to 0.5 - 2uM as in cooper 2012
    %      A simulated
    % HRF was then designed that consisted of a gamma function with
    % a time-to-peak of 7 s, a duration of 20 s and an amplitude defined
    % so as to produce a 15µM increase in HbO concentration and a
    % 5µM decrease in HbR concentration [these figures include a partial volume correction factor of 50 (Strangman et al., 2003)]. S
    % so I used 10~20µM increase
    amp_HbO = 10 + (20 - 10) * rand;
    HRFs    =   make_HRFs(s, amp_HbO);
    HRFs_HbO = HRFs.HbO;
    HRFs_HbR = HRFs.HbR;
    %% sum up
    noised_HRF_HbO = Resting_HbO + HRFs_HbO + noise_HbO;
    noised_HRF_HbR = Resting_HbR + HRFs_HbR + noise_HbR;
    %% plot example
    if i == loc_example
        save('Processed_data/Example.mat','HRFs_HbO','noise_HbO','noised_HRF_HbO','Resting_HbO')
        plot_sim_data(HRFs_HbO,noise_HbO,noised_HRF_HbO,Resting_HbO)
    end
    %% save the data
    noised_HRF_HbO_matrix(i,:)     =   noised_HRF_HbO;
    noise_HbO_matrix(i,:)          =   noise_HbO;
    HRF_profile_HbO_matrix(i,:)    =   HRFs_HbO;
    
    noised_HRF_HbR_matrix(i,:)     =   noised_HRF_HbR;
    noise_HbR_matrix(i,:)          =   noise_HbR;
    HRF_profile_HbR_matrix(i,:)    =   HRFs_HbR;
end

%% train, val, test 8:1:1
p           =   randperm(n_sample);
n_train     =   round(n_sample * 0.8);
n_val       =   round(n_sample * 0.1);
train_idx   =   p(1:n_train);
val_idx     =   p(n_train+1:n_train+n_val);
test_idx    =   p(n_train+n_val+1:end);

HRF_train   =   [HRF_profile_HbO_matrix(train_idx,:);  HRF_profile_HbR_matrix(train_idx,:)];
HRF_val     =   [HRF_profile_HbO_matrix(val_idx,:);    HRF_profile_HbR_matrix(val_idx,:)];
HRF_test    =   [HRF_profile_HbO_matrix(test_idx,:);   HRF_profile_HbR_matrix(test_idx,:)];

HRF_train_noised    =   [noised_HRF_HbO_matrix(train_idx,:);   noised_HRF_HbR_matrix(train_idx,:)];
HRF_val_noised      =   [noised_HRF_HbO_matrix(val_idx,:);     noised_HRF_HbR_matrix(val_idx,:)];
HRF_test_noised     =   [noised_HRF_HbO_matrix(test_idx,:);    noised_HRF_HbR_matrix(test_idx,:)];

Noise_train     =   [noise_HbO_matrix(train_idx,:);    noise_HbR_matrix(train_idx,:)];
Noise_val       =   [noise_HbO_matrix(val_idx,:);      noise_HbR_matrix(val_idx,:)];
Noise_test      =   [noise_HbO_matrix(test_idx,:);     noise_HbR_matrix(test_idx,:)];

Resting_train   =   [simulated_HbO(train_idx,:);  simulated_HbR(train_idx,:)];
Resting_val     =   [simulated_HbO(val_idx,:);    simulated_HbR(val_idx,:)];
Resting_test    =   [simulated_HbO(test_idx,:);   simulated_HbR(test_idx,:)];

%% save
save('Processed_data/SimulateData.mat',...
    'HRF_train',...
    'HRF_val',...
    'HRF_test',...
    'HRF_train_noised',...
    'HRF_val_noised',...
    'HRF_test_noised');
%noise
save('Processed_data/Noise.mat','Noise_train','Noise_val','Noise_test')
% Resting
save('Processed_data/Resting.mat','Resting_train','Resting_val','Resting_test')
% number of noise
save('Processed_data/n_MA_list.mat','peak_n_list','shift_n_list')

%% sub functions
function noise = add_peak(noise,diff_HbO_list,t0)
global fs_new
tp      =   length(noise);
A       =   diff_HbO_list(randi(length(diff_HbO_list)));
b_low   =   0;b_high = 1.5;
b       =   b_low + rand * (b_high - b_low);
neg_sign_values = [-1 1]; % whether the MA is upwards or downwards
neg_sign        = neg_sign_values(randi(2));
t       =   (1:tp)/fs_new;
noise   =   noise + neg_sign *A.*exp(-abs(t-t0./fs_new)./(b));
end
function noise = add_shift(noise,diff_HbO_list,start_point,transition)
DC_shift    =   diff_HbO_list(randi(length(diff_HbO_list)));
end_point   =   start_point + transition;
shift_sim   =   zeros(size(noise));
shift_sim(start_point:end_point) = linspace(0,DC_shift,end_point - start_point + 1);
shift_sim(end_point:end) = DC_shift;
neg_sign_values = [-1 1]; % whether the MA is upwards or downwards
neg_sign        = neg_sign_values(randi(2));
noise       =   noise + neg_sign * shift_sim;
end