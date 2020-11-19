clear all
close all

%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define global variables
global fs_new
fs_new = 7;
global rt
global pt
rt =   40; %resting time 5.7s
pt =   512-40; %performance time 65.536s - 5.7s
STD = 10;
%% load fNIRS file

DataDir =   '../New Datasets/05/motion_artifacts_1';
Files   =   dir(fullfile(DataDir,'*.nirs'));
n_files =   length(Files);

%% define the variables to be calculated

% save No. of MAs for each files n_files X [n_MA_no_correction,n_MA_PCA99,
% dc_avg_PCA79,n_MA_Spline,n_MA_Wavelet01,]
MA_matrix    =   zeros(n_files,7);

for file = 1:n_files
    name = Files(file).name;
    fprintf('fnirs file is %s\n',name);
    fNIRS_data = load([DataDir,'/',name],'-mat');
    %% unify the sampling rate to around 7.8 Hz
    t           =   fNIRS_data.t;
    fs          =   1/(t(2)-t(1)); % old sampling rate
    fs_new      =   7;
    fsn         =   fs/fs_new;
    fNIRS_data  =   Downsample(fNIRS_data, fsn);
    %% define variables
    d           =   fNIRS_data.d;
    n_Ch        =   size(d,2)/2;
    t           =   fNIRS_data.t;
    tIncMan     =   ones(size(t));
    SD          =   fNIRS_data.SD;
    SD          =   enPruneChannels(d,SD,tIncMan,[1e+04 1e+07],2,[0  45],0);
    s           =   zeros(1,length(t));
    s((rt+1):512:length(t)-512) = 1;
    %% standard processing wo correction
    dod             =   hmrIntensity2OD(d);
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
    dod             =   hmrBandpassFilt(dod,t,0,0.5);
    dc              =   hmrOD2Conc(dod,SD,[6  6]);
    %% add HRFs
    % the amplitude of HRF is 0.04uMol (0.01-0.1) in Barker 2013,
    amp_HbO         =   0.04;
    HRFs            =   make_HRFs(s, amp_HbO);
    dc_act          =   zeros(size(dc));
    dc_act(:,1,:)   =   squeeze(dc(:,1,:)) + repmat(HRFs.HbO',1,n_Ch);
    dc_act(:,2,:)   =   squeeze(dc(:,2,:)) + repmat(HRFs.HbR',1,n_Ch);
    dc_act(:,3,:)   =   dc_act(:,1,:) + dc_act(:,2,:);
    % derive two sets: dc: with no activation, dc_act: with activation
    %% no correction
    [dc_no_crct, ~, ~, ~, ~, ~] = hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] );
    [dc_act_no_crct, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_act, s', t, [-39/fs_new (512-40)/fs_new] );
    n_MA_no_crct = count_no_correction(tIncChAuto);
    %% standard processing with PCA99
    sigma           =   0.99;
    [dc_PCA99,~]                 =   proc_PCA(dc,s,SD,t,tIncMan,STD,sigma);
    [dc_act_PCA99,n_MA_PCA99]    =   proc_PCA(dc_act,s,SD,t,tIncMan,STD,sigma);
    %% standard processing with PCA79
    sigma           =   0.79;
    [dc_PCA79,~]                 =   proc_PCA(dc,s,SD,t,tIncMan,STD,sigma);
    [dc_act_PCA79,n_MA_PCA79]    =   proc_PCA(dc_act,s,SD,t,tIncMan,STD,sigma);
    %% standard processing with Spline 99
    p               =   0.99;
    [dc_Spline,~]                   =   proc_Spline(dc,s,SD,t,tIncMan,STD,p);
    [dc_act_Spline,n_MA_Spline]     =   proc_Spline(dc_act,s,SD,t,tIncMan,STD,p);
    %% standard processing with Wavelet01
    alpha                   =   0.1;
    [dc_Wavelet,~]                    =   proc_Wavelet(dc,s,SD,t,tIncMan,STD,alpha);
    [dc_act_Wavelet,n_MA_Wavelet]     =   proc_Wavelet(dc_act,s,SD,t,tIncMan,STD,alpha);
    %% hmrKalman
    [dc_Kalman,~]                     =   proc_Kalman(dc,s,SD,t,tIncMan,STD);
    [dc_act_Kalman,n_MA_Kalman]       =   proc_Kalman(dc_act,s,SD,t,tIncMan,STD);
    %% hmrCbsi
    [dc_Cbsi,~]                     =   proc_Cbsi(dc,s,SD,t,tIncMan,STD);
    [dc_act_Cbsi,n_MA_Cbsi]         =   proc_Cbsi(dc_act,s,SD,t,tIncMan,STD);
    %% write the processed data
    MA_matrix(file,:)               =   [n_MA_no_crct,n_MA_PCA99,n_MA_PCA79,n_MA_Spline,n_MA_Wavelet,n_MA_Kalman,n_MA_Cbsi];
    net_input(file).dc_act          =   dc_act;
    net_input(file).dc              =   dc;
    Proc_data(file).HRF             =   HRFs;
    Proc_data(file).dc_no_crct      =   dc_no_crct;
    Proc_data(file).dc_PCA99        =   dc_PCA99;
    Proc_data(file).dc_PCA79        =   dc_PCA79;
    Proc_data(file).dc_Spline       =   dc_Spline;
    Proc_data(file).dc_Wavelet      =   dc_Wavelet;
    Proc_data(file).dc_Kalman       =   dc_Kalman;
    Proc_data(file).dc_Cbsi         =   dc_Cbsi;
    Proc_data(file).dc_act_no_crct  =   dc_act_no_crct;
    Proc_data(file).dc_act_PCA99    =   dc_act_PCA99;
    Proc_data(file).dc_act_PCA79    =   dc_act_PCA79;
    Proc_data(file).dc_act_Spline   =   dc_act_Spline;
    Proc_data(file).dc_act_Wavelet  =   dc_act_Wavelet;
    Proc_data(file).dc_act_Kalman   =   dc_act_Kalman;
    Proc_data(file).dc_act_Cbsi     =   dc_act_Cbsi;
end
%% save processed data
save('Processed_data/Process_real_data.mat','Proc_data')
save('Processed_data/RealData.mat','net_input')
