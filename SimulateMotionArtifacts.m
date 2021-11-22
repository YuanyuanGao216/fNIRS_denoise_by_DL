clear all
% close all

DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
%     loc_example = -1;
    %% load data
    % here the loaded simulated Hb are from that person. So we concat the
    % ones except subfolder. But list is already from the others
    folders = subfolders;
    folders(subfolder) = [];
    simulated_HbO_mat = [];
    simulated_HbR_mat = [];
    for f = 1:length(folders)
        sim_Resting_path = fullfile(DataDir, folders(f).name, 'sim_Resting.mat');
        load(sim_Resting_path, 'simulated_HbO', 'simulated_HbR', 'loc_example')
        simulated_HbO_mat = [simulated_HbO_mat; simulated_HbO];
        simulated_HbR_mat = [simulated_HbR_mat; simulated_HbR];
    end
    simulated_HbO = simulated_HbO_mat;
    simulated_HbR = simulated_HbR_mat;
    clear simulated_HbO_mat simulated_HbR_mat
    
    list_path = fullfile(DataDir, subfolders(subfolder).name, 'list.mat');
    load(list_path, 'diff_HbO_list', 'diff_HbR_list', 'd_MA_list');
    
    % make samples equal in HbO and HbR
%     n_HbO = size(simulated_HbO,1);
%     n_HbR = size(simulated_HbR,1);
%     n_sample = min(n_HbO,n_HbR);
%     simulated_HbO = simulated_HbO(1:n_sample,:);
%     simulated_HbR = simulated_HbR(1:n_sample,:);
    %% define constants
    define_constants

    %% define n_list: a list of No. of MA for simulated samples 
    tp = size(simulated_HbO,2);
    n_sample = size(simulated_HbO,1);
    t = (1:tp)/fs_new;
    n_MA_list = round(d_MA_list * tp);
    n_list = n_MA_list(randi(length(n_MA_list),n_sample,1));
    n_list = n_list';
    % they are either peak or shift
    peak_n_list = zeros(size(n_list));
    shift_n_list = zeros(size(n_list));
    for i = 1:length(n_list)
        shift_n_list(i) = round(randi(n_list(i))/3);
        peak_n_list(i) =  n_list(i) - shift_n_list(i);
    end

    %%
    s = zeros(1,tp);
    s((rt):512:tp) = 1;

    %% HbO
    noised_HRF_HbO_matrix = zeros(size(simulated_HbO));
    noised_no_HRF_HbO_matrix = zeros(1,size(simulated_HbO,2));
    noise_HbO_matrix = zeros(size(simulated_HbO));
    HRF_profile_HbO_matrix = zeros(size(simulated_HbO));
    HRF_profile_HbO_matrix_no_HRF = zeros(1,size(simulated_HbO,2));
    simulated_HbO_no_HRF = zeros(1,size(simulated_HbO,2));
    noise_HbO_matrix_no_HRF = zeros(1,size(simulated_HbO,2));

    noised_HRF_HbR_matrix = zeros(size(simulated_HbR));
    noised_no_HRF_HbR_matrix = zeros(1,size(simulated_HbO,2));
    noise_HbR_matrix = zeros(size(simulated_HbR));
    HRF_profile_HbR_matrix = zeros(size(simulated_HbR));
    HRF_profile_HbR_matrix_no_HRF = zeros(1,size(simulated_HbO,2));
    simulated_HbR_no_HRF = zeros(1,size(simulated_HbO,2));
    noise_HbR_matrix_no_HRF = zeros(1,size(simulated_HbO,2));
    peak_n_list_new = [];
    shift_n_list_new = [];
    k = 1;
    
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
        % change it to 0~20
        % even though from 0, it doesn't perform good on 0 activations,
        % need to add more 0 samples
        amp_HbO = 10 + (20 - 10) * rand; % 10-20
        time_to_peak = 1 + (11 - 1)*rand; % 1-11
        duration = 15 + (30 - 15)*rand; % 1-30
        sigma = 5 + (15 - 5)*rand;
%         amp_HbO = 20; % 10-20
%         time_to_peak = 1; % 1-11
%         duration = 15; % 1-30
%         sigma = 15;
        HRFs    =   make_HRFs(s, amp_HbO, time_to_peak, duration, sigma);
        
        HRFs_HbO = HRFs.HbO;
        HRFs_HbR = HRFs.HbR;
%         figure; subplot(211);plot(HRFs_HbO);subplot(212);plot(HRFs_HbR)
        %% sum up
        noised_HRF_HbO = Resting_HbO + HRFs_HbO + noise_HbO;
        noised_HRF_HbR = Resting_HbR + HRFs_HbR + noise_HbR;
        
        %% plot example
        if i == loc_example && subfolder == 1
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
        
        
        
        %% for zeros training
        if mod(i,10) == 0
            noised_no_HRF_HbO = Resting_HbO + noise_HbO;
            noised_no_HRF_HbR = Resting_HbR + noise_HbR;
            noised_no_HRF_HbO_matrix(k,:)  =   noised_no_HRF_HbO;
            noised_no_HRF_HbR_matrix(k,:)  =   noised_no_HRF_HbR;
            HRF_profile_HbO_matrix_no_HRF(k,:) = zeros(size(noised_no_HRF_HbO));
            HRF_profile_HbR_matrix_no_HRF(k,:) = zeros(size(noised_no_HRF_HbR));
            simulated_HbO_no_HRF(k,:) = Resting_HbO;
            simulated_HbR_no_HRF(k,:) = Resting_HbR;
            noise_HbO_matrix_no_HRF(k, :) = noise_HbO;
            noise_HbR_matrix_no_HRF(k, :) = noise_HbR;
            peak_n_list_new = [peak_n_list_new, n_peak];
            shift_n_list_new = [shift_n_list_new, n_shift];
            k = k + 1;
        end
        %% also want to add real_reasting + synthetic HRFs to the training data
        
    end
    noised_HRF_HbO_matrix   = [noised_HRF_HbO_matrix; noised_no_HRF_HbO_matrix];
    noised_HRF_HbR_matrix   = [noised_HRF_HbR_matrix; noised_no_HRF_HbR_matrix];
    HRF_profile_HbO_matrix  = [HRF_profile_HbO_matrix; HRF_profile_HbO_matrix_no_HRF];
    HRF_profile_HbR_matrix  = [HRF_profile_HbR_matrix; HRF_profile_HbR_matrix_no_HRF];
    simulated_HbO           = [simulated_HbO; simulated_HbO_no_HRF];
    simulated_HbR           = [simulated_HbR; simulated_HbR_no_HRF];
    noise_HbO_matrix        = [noise_HbO_matrix; noise_HbO_matrix_no_HRF];
    noise_HbR_matrix        = [noise_HbR_matrix; noise_HbR_matrix_no_HRF];
    peak_n_list             = [peak_n_list, peak_n_list_new];
    shift_n_list            = [shift_n_list, shift_n_list_new];
    %% train, val, test 8:1:1
    n_sample = size(noised_HRF_HbO_matrix,1);
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
    savepath = fullfile(DataDir, subfolders(subfolder).name, 'SimulateData.mat');
    save(savepath,...
        'HRF_train',...
        'HRF_val',...
        'HRF_test',...
        'HRF_train_noised',...
        'HRF_val_noised',...
        'HRF_test_noised');
    %noise
    savepath = fullfile(DataDir, subfolders(subfolder).name, 'Noise.mat');
    save(savepath,'Noise_train','Noise_val','Noise_test')
    % Resting
    savepath = fullfile(DataDir, subfolders(subfolder).name, 'Resting.mat');
    save(savepath,'Resting_train','Resting_val','Resting_test')
    % number of noise
    savepath = fullfile(DataDir, subfolders(subfolder).name, 'n_MA_list.mat');
    save(savepath,'peak_n_list','shift_n_list','p')
end

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