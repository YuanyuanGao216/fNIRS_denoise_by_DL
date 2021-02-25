clear all
close all

%% add homer path
pathHomer = 'Tools/homer2_src_v2_3_10202017/';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define global variables
define_constants

%% load fNIRS file

DataDir =   'Data/';
Files   =   dir(fullfile(DataDir,'*.nirs'));
n_files =   length(Files);

%% define the variables to be calculated

% save No. of MAs for each files n_files X [n_MA_no_correction,dc_avg_PCA97,n_MA_Spline99,n_MA_Wavelet01,]
MA_matrix    =   zeros(n_files,7);

for file = 1:n_files
    name = Files(file).name;
    fprintf('fnirs file is %s\n',name);
    fNIRS_data = load([DataDir,'/',name],'-mat');
    %% unify the sampling rate to around 7.8 Hz
    fNIRS_data  =   Downsample(fNIRS_data);
    %% define variables
    d           =   fNIRS_data.d;
    SD          =   fNIRS_data.SD;
    t           =   fNIRS_data.t;
    fs          =   1 / (t(2) - t(1));
    fprintf('fs is %f\n',fs);
    tIncMan     =   ones(size(t));
    n_Ch        =   size(d,2)/2;
    
    s           =   zeros(1,length(t));
    s((rt+1):512:length(t)-512) = 1;
    %% standard processing wo correction
    [dc,SD,tInc] = proc_wo_crct(d,SD,t,STD, OD_thred);
    
    %% add HRFs
    % % the amplitude of HRF is 0.04uMol (0.01-0.1) in Barker 2013, 2 uM in cooper 2012
    % 0.04uMol is too small, here I temporily set amp to 0.5 - 2uM as in cooper 2012
    %      A simulated
    % HRF was then designed that consisted of a gamma function with
    % a time-to-peak of 7 s, a duration of 20 s and an amplitude defined
    % so as to produce a 15µM increase in HbO concentration and a
    % 5µM decrease in HbR concentration [these figures include a partial volume correction factor of 50 (Strangman et al., 2003)]. S
    % so I used 10~20µM increase
    amp_HbO         =   15;
    HRFs            =   make_HRFs(s, amp_HbO);
    dc_act          =   zeros(size(dc));
    dc_act(:,1,:)   =   squeeze(dc(:,1,:)) + repmat(HRFs.HbO',1,n_Ch);
    dc_act(:,2,:)   =   squeeze(dc(:,2,:)) + repmat(HRFs.HbR',1,n_Ch);
    dc_act(:,3,:)   =   dc_act(:,1,:) + dc_act(:,2,:);
    % derive two sets: dc: with no activation, dc_act: with activation
    %% no correction
    [dc_no_crct, ~, ~, ~, ~, ~] = hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] );
    [dc_act_no_crct, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_act, s', t, [-39/fs_new (512-40)/fs_new] );
    n_MA_no_crct = count_MA(tInc);
    %% standard processing with PCA97
    sigma           =   0.97;
    [dc_PCA97,n_MA_PCA97]                 =   proc_PCA(dc,s,SD,t,tIncMan,STD,OD_thred,sigma);
    [dc_act_PCA97,~]    =   proc_PCA(dc_act,s,SD,t,tIncMan,STD,OD_thred,sigma);
    %% standard processing with PCA99
    sigma           =   0.99;
    [dc_PCA99,n_MA_PCA99]                 =   proc_PCA(dc,s,SD,t,tIncMan,STD,OD_thred,sigma);
    [dc_act_PCA99,~]    =   proc_PCA(dc_act,s,SD,t,tIncMan,STD,OD_thred,sigma);
    %% standard processing with Spline 99
    p               =   0.99;
    [dc_Spline,n_MA_Spline]                   =   proc_Spline(dc,s,SD,t,tIncMan,STD,OD_thred,p);
    [dc_act_Spline,~]     =   proc_Spline(dc_act,s,SD,t,tIncMan,STD,OD_thred,p);
    %% standard processing with Wavelet01
    alpha                   =   0.1;
    [dc_Wavelet,n_MA_Wavelet]                    =   proc_Wavelet(dc,s,SD,t,tIncMan,STD,OD_thred,alpha);
    [dc_act_Wavelet,~]     =   proc_Wavelet(dc_act,s,SD,t,tIncMan,STD,OD_thred,alpha);
    %% hmrKalman
    [dc_Kalman,n_MA_Kalman]                     =   proc_Kalman(dc,s,SD,t,tIncMan,OD_thred,STD);
    [dc_act_Kalman,~]       =   proc_Kalman(dc_act,s,SD,t,tIncMan,OD_thred,STD);
    %% hmrCbsi
    [dc_Cbsi,n_MA_Cbsi]                     =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD);
    [dc_act_Cbsi,~]         =   proc_Cbsi(dc_act,s,SD,t,tIncMan,OD_thred,STD);
    %% write the processed data
    MA_matrix(file,:)               =   [n_MA_no_crct, n_MA_Spline,n_MA_Wavelet,n_MA_Kalman, n_MA_PCA99,n_MA_PCA97,n_MA_Cbsi];
    net_input(file).dc_act          =   dc_act;
    net_input(file).dc              =   dc;
    Proc_data(file).HRF             =   HRFs;
    Proc_data(file).dc_no_crct      =   dc_no_crct;
    Proc_data(file).dc_PCA97        =   dc_PCA97;
    Proc_data(file).dc_PCA99        =   dc_PCA99;
    Proc_data(file).dc_Spline       =   dc_Spline;
    Proc_data(file).dc_Wavelet      =   dc_Wavelet;
    Proc_data(file).dc_Kalman       =   dc_Kalman;
    Proc_data(file).dc_Cbsi         =   dc_Cbsi;
    Proc_data(file).dc_act_no_crct  =   dc_act_no_crct;
    Proc_data(file).dc_act_PCA97    =   dc_act_PCA97;
    Proc_data(file).dc_act_PCA99    =   dc_act_PCA99;
    Proc_data(file).dc_act_Spline   =   dc_act_Spline;
    Proc_data(file).dc_act_Wavelet  =   dc_act_Wavelet;
    Proc_data(file).dc_act_Kalman   =   dc_act_Kalman;
    Proc_data(file).dc_act_Cbsi     =   dc_act_Cbsi;
end
%% save processed data
save('Processed_data/Process_real_data.mat','Proc_data','MA_matrix')
save('Processed_data/RealData.mat','net_input')
