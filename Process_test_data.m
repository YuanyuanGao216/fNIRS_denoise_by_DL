%% add homer path
pathHomer = 'Tools/homer2_src_v2_3_10202017/';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

define_constants
%% load data load each LOO dataset
DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
    sim_path = fullfile(DataDir, subfolders(subfolder).name, 'SimulateData.mat');
    load(sim_path,'HRF_test_noised')

    [m,n] = size(HRF_test_noised);
    HbO_noised = HRF_test_noised(1:m/2,:);
    HbR_noised = HRF_test_noised(m/2+1:end,:);
    %% define variables 
    define_constants

    %% 
    SD1.MeasList = [1,1,1,1;1,1,1,2];
    SD1.MeasListAct = [1 1];
    SD1.Lambda = [760;850];
    SD1.SrcPos = [-2.9017 10.2470 -0.4494];
    SD1.DetPos = [-4.5144 9.0228 -1.6928];
    ppf = [6,6];
    t  = 1/fs_new:1/fs_new:size(HbO_noised,2)/fs_new;
    s  = zeros(1,length(t));
    s((rt):512:length(t)) = 1;
    tIncMan=ones(size(t))';
    SD = SD1;
    %% Cbsi
    HbO_Cbsi    = [];
    HbR_Cbsi    = [];
    n_Cbsi = 0;
    T_Cbsi = 0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD);
        T_Cbsi          =   T_Cbsi + toc;

        HbO_Cbsi(end+1,:) = dc_avg(:,1)';
        HbR_Cbsi(end+1,:) = dc_avg(:,2)';
        n_Cbsi          =   n_Cbsi + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Cbsi.mat');
    save(save_path,'HbO_Cbsi','HbR_Cbsi','n_Cbsi', 'T_Cbsi')

    %% PCA97
    HbO_PCA97   = [];
    HbR_PCA97   = [];
    n_PCA97 = 0;
    sigma   =  0.97;
    T_PCA97 =   0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_PCA(dc, s, SD, t, tIncMan,STD, OD_thred, sigma);
        T_PCA97         =   T_PCA97 + toc;

        HbO_PCA97(end+1,:)  = dc_avg(:,1)';
        HbR_PCA97(end+1,:)  = dc_avg(:,2)';
        n_PCA97             =   n_PCA97 + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_PCA97.mat');
    save(save_path, 'HbO_PCA97', 'HbR_PCA97', 'n_PCA97', 'T_PCA97')
%     %% PCA99
%     HbO_PCA99   = [];
%     HbR_PCA99   = [];
%     n_PCA99 = 0;
%     sigma   =  0.99;
%     T_PCA99 =   0;
% 
%     for i = 1:m/2
%         dc_HbO          =   HbO_noised(i,:);
%         dc_HbR          =   HbR_noised(i,:);
%         dc              =   [dc_HbO;dc_HbR]';
% 
%         tic
%         [dc_avg,n_MA]   =   proc_PCA(dc, s, SD, t, tIncMan,STD, OD_thred, sigma);
%         T_PCA99         =   T_PCA99 + toc;
% 
%         HbO_PCA99(end+1,:)  = dc_avg(:,1)';
%         HbR_PCA99(end+1,:)  = dc_avg(:,2)';
%         n_PCA99             =   n_PCA99 + n_MA;
%     end
%     save('Processed_data/Testing_PCA99.mat','HbO_PCA99','HbR_PCA99','n_PCA99')
    %% Kalman
    HbO_Kalman  = [];
    HbR_Kalman  = [];
    n_Kalman = 0;
    T_Kalman = 0;

    for i = 1:m/2
        dc_HbO          =   HbO_noised(i,:);
        dc_HbR          =   HbR_noised(i,:);
        dc              =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]   =   proc_Kalman(dc,s,SD,t,tIncMan,OD_thred,STD);
        T_Kalman        =   T_Kalman + toc;

        HbO_Kalman(end+1,:)  = dc_avg(:,1)';
        HbR_Kalman(end+1,:)  = dc_avg(:,2)';
        n_Kalman             =   n_Kalman + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Kalman.mat');
    save(save_path, 'HbO_Kalman', 'HbR_Kalman', 'n_Kalman', 'T_Kalman')
    %% Spline
    HbO_Spline  = [];
    HbR_Spline  = [];
    n_Spline = 0;
    T_Spline =   0;
    p       =   0.99;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]       =   proc_Spline(dc, s, SD1, t, tIncMan, STD, OD_thred,p);
        T_Spline            =   T_Spline + toc;

        HbO_Spline(end+1,:)  = dc_avg(:,1)';
        HbR_Spline(end+1,:)  = dc_avg(:,2)';
        n_Spline             =   n_Spline + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Spline.mat');
    save(save_path, 'HbO_Spline', 'HbR_Spline', 'n_Spline', 'T_Spline')

    %% Wavelet01
    HbO_Wavelet01 = [];
    HbR_Wavelet01 = [];
    n_Wavelet01 = 0;
    T_Wavelet01 =   0;
    iqr       =   0.75;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';

        tic
        [dc_avg,n_MA]       =   proc_Wavelet(dc,s,SD1,t,tIncMan,STD,OD_thred,iqr);
        T_Wavelet01         =   T_Wavelet01 + toc;

        HbO_Wavelet01(end+1,:)  = dc_avg(:,1)';
        HbR_Wavelet01(end+1,:)  = dc_avg(:,2)';
        n_Wavelet01             =   n_Wavelet01 + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Wavelet01.mat');
    save(save_path,'HbO_Wavelet01','HbR_Wavelet01','n_Wavelet01', 'T_Wavelet01')

    %% no correction
    n_no_correction = 0;

    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:);
        dc_HbR              =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';
        dod                 =   hmrConc2OD( dc, SD1, ppf );
        [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,OD_thred);
        n_MA                =   count_MA(tIncAuto);

        n_no_correction     =   n_no_correction + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_no_correction.mat');
    save(save_path,'n_no_correction')
    %% NN count noise
    NN_path = fullfile(DataDir, subfolders(subfolder).name, 'Test_NN_8layers.mat');

    load(NN_path)% File exists.
    Hb_NN = Y_test;
    m = size(Hb_NN,1);

    HbO_NN = Hb_NN(:,1:512);
    HbR_NN = Hb_NN(:,512+1:end);
    % HbO_NN = reshape(HbO_NN',[512*5,1684]);
    % HbR_NN = reshape(HbR_NN',[512*5,1684]);
    % HbO_NN = HbO_NN';
    % HbR_NN = HbR_NN';
    n_NN = 0;
    t = t(1:512);
    tIncMan = tIncMan(1:512);

    for i = 1:m
        dc_HbO  =   HbO_NN(i,:);
        dc_HbR  =   HbR_NN(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        dod     =   hmrConc2OD( dc, SD1, ppf );
        [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
    %     tIncAuto=   hmrMotionArtifact(dod,t,SD1,tIncMan,0.5,1,STD,OD_thred);
        n_MA    =   count_MA(tIncAuto);

        n_NN = n_NN + n_MA;
    end
    save_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_NN.mat');
    save(save_path,'n_NN')

end