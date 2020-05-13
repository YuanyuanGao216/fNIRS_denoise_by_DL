%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define variables to be calulate
n_MA_HbO_list = [];
n_MA_HbR_list = [];
Real_HbO    = [];
Real_HbR    = [];
HbO_Spline  = [];
HbR_Spline  = [];
HbO_Wavelet05 = [];
HbR_Wavelet05 = [];
HbO_Wavelet35 = [];
HbR_Wavelet35 = [];
HbO_Kalman  = [];
HbR_Kalman  = [];
HbO_PCA99   = [];
HbR_PCA99   = [];
HbO_PCA50   = [];
HbR_PCA50   = [];
HbO_Cbsi    = [];
HbR_Cbsi    = [];

n_Spline_HbO = 0;
n_Spline_HbR = 0;
n_Wavelet05_HbO = 0;
n_Wavelet05_HbR = 0;
n_Wavelet35_HbO = 0;
n_Wavelet35_HbR = 0;
n_Kalman_HbO = 0;
n_Kalman_HbR = 0;
n_PCA99_HbO = 0;
n_PCA99_HbR = 0;
n_PCA50_HbO = 0;
n_PCA50_HbR = 0;
n_Cbsi_HbO = 0;
n_Cbsi_HbR = 0;

STD = 10;
%% load fNIRS file
% loop subfolders and files
if ispc
    DataDir = '..\..\Buffalo_study\Raw_Data\Study_#2';
else
    DataDir = '../../Buffalo_study/Raw_Data/Study_#2';
end
if ispc
    start_point = 3;
else
    start_point =4;
end
Subfolders = dir(DataDir);
for folder = start_point:length(Subfolders)
    name = Subfolders(folder).name;
    if strcmp(name,'L8') % L8 dropped out of the study and his data is not complete
        continue
    end
    fprintf('subject is %s\n',name);
    SubDir = [DataDir,'/',name];
    Files = dir(fullfile(SubDir,'*.nirs'));
    for file = 1:length(Files)
        filename = Files(file).name;
        fprintf('fnirs file is %s\n',filename);
        fNIRS_data = load([SubDir,'/',filename],'-mat');
        %% define variables
        d           =   fNIRS_data.d;
        SD          =   fNIRS_data.SD;
        t           =   fNIRS_data.t;
        tIncMan     =   ones(size(t));
        
        if size(fNIRS_data.s,2) ~= 1
            fNIRS_data.s = sum(fNIRS_data.s,2);
        end
        s           =   fNIRS_data.s;
        %% standard processing wo correction
        SD              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod             =   hmrIntensity2OD(d);
        [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
        dod             =   hmrBandpassFilt(dod,t,0,0.5);
        dc_nocorrection              =   hmrOD2Conc(dod,SD,[6  6]);
        %% standard processing with PCA99
        SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                     =   hmrIntensity2OD(d);
        [dod_PCA99,~,~,~,~]     =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,STD,200,0.99,5);
        [~,tIncAuto_PCA99]      =   hmrMotionArtifactByChannel(dod_PCA99,t,SD,tIncMan,0.5,1,STD,200);
        dod_PCA99               =   hmrBandpassFilt(dod_PCA99,t,0,0.5);
        dc_PCA99                =   hmrOD2Conc(dod_PCA99,SD,[6  6]);
        %% standard processing with PCA50
        SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                     =   hmrIntensity2OD(d);
        [dod_PCA50,~,~,~,~]     =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,STD,200,0.50,5);
        [~, tIncAuto_PCA50]     =   hmrMotionArtifactByChannel(dod_PCA50,t,SD,tIncMan,0.5,1,STD,200);
        dod_PCA50               =   hmrBandpassFilt(dod_PCA50,t,0,0.5);
        dc_PCA50                =   hmrOD2Conc(dod_PCA50,SD,[6  6]);
        %% standard processing with Spline 99
        SD                              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                             =   hmrIntensity2OD(d);
        [~,tIncChAuto_before_Spline]    =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
        [dod_Spline]                    =   hmrMotionCorrectSpline(dod,t,SD,tIncChAuto_before_Spline,0.99);
        [~,tIncChAuto_after_Spline]     =   hmrMotionArtifactByChannel(dod_Spline,t,SD,tIncMan,0.5,1,STD,200);
        dod_Spline                      =   hmrBandpassFilt(dod_Spline,t,0,0.5);
        dc_Spline                       =   hmrOD2Conc(dod_Spline,SD,[6  6]);
        %% standard processing with Wavelet05
        SD                  =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0); 
        dod                 =   hmrIntensity2OD(d);
        [dod_Wavelet05]       =   hmrMotionCorrectWavelet(dod,SD,0.05);
        [~,tIncAuto_Wavelet05]=   hmrMotionArtifactByChannel(dod_Wavelet05,t,SD,tIncMan,0.5,1,STD,200);
        dod_Wavelet05         =   hmrBandpassFilt(dod_Wavelet05,t,0,0.5);
        dc_Wavelet05          =   hmrOD2Conc(dod_Wavelet05,SD,[6  6]);
        %% standard processing with Wavelet35
        SD                  =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0); 
        dod                 =   hmrIntensity2OD(d);
        [dod_Wavelet35]       =   hmrMotionCorrectWavelet(dod,SD,0.35);
        [~,tIncAuto_Wavelet35]=   hmrMotionArtifactByChannel(dod_Wavelet35,t,SD,tIncMan,0.5,1,STD,200);
        dod_Wavelet35         =   hmrBandpassFilt(dod_Wavelet35,t,0,0.5);
        dc_Wavelet35          =   hmrOD2Conc(dod_Wavelet35,SD,[6  6]);
        %% hmrKalman
        SD              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod             =   hmrIntensity2OD(d);
        y               =   dod;
        oscFreq         =   [0,0.01,0.001,0.0001];
        xo              =   ones(1,length(oscFreq)+1)*y(1,1);
        Po              =   ones(1,length(oscFreq)+1)*(y(1,1)^2);
        Qo              =   zeros(1,length(oscFreq)+1);
        hrfParam        =   [2 2];
        [x, yStim,dod_Kalman,y,C,Q] = hmrKalman2( y, s, t, xo, Po, Qo, 'box', hrfParam, oscFreq );
        [~,tIncAuto_Kalman]   =   hmrMotionArtifactByChannel(dod_Kalman,t,SD,tIncMan,0.5,1,STD,200);
        dod_Kalman          =   hmrBandpassFilt(dod_Kalman,t,0,0.5);
        dc_Kalman           =   hmrOD2Conc(dod_Kalman,SD,[6  6]);
        %% hmrCbsi
        SD = enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod = hmrIntensity2OD(d);
        dod = hmrBandpassFilt(dod,t,0,0.5);
        dc = hmrOD2Conc(dod,SD,[6  6]);
        [dc_Cbsi] = hmrMotionCorrectCbsi(dc,SD,0);
        dod_Cbsi = hmrConc2OD(dc_Cbsi,SD,[6  6]);
        [~,tIncAuto_Cbsi] = hmrMotionArtifactByChannel(dod_Cbsi,t,SD,tIncMan,0.5,1,STD,200);
        %%
        % select first 65s (512 points) period within trials
        if size(fNIRS_data.s,2) ~= 1
            fNIRS_data.s = sum(fNIRS_data.s,2);
        end
        stimuli_list = find(fNIRS_data.s==1);
        n_stim = length(stimuli_list);

        % 0 in tIncAuto means artifact
        n_Ch        =   size(dod,2);
        Ch_list     =   1:n_Ch;
        Ch_short    =   [3,6,9,15,20,25,31,36]';
        Ch_short    =   [Ch_short;Ch_short+36];
        Ch_Prune    =   find(SD.MeasListAct==0);
        Ch_list([Ch_short;Ch_Prune]) = [];
        for i = 1:2:n_stim
            stim_start   = stimuli_list(i);
            stim_end     = stimuli_list(i+1);
            if stim_end - stim_start < 512
                continue
            end
            for Ch = Ch_list
                [n_MA,runstarts,runends] = CalMotionArtifact(tIncChAuto(stim_start:512,Ch));
                if n_MA == 0
                    continue
                end
                if Ch < 36
                    n_MA_HbO_list(end+1) = n_MA;
                else
                    n_MA_HbR_list(end+1) = n_MA;
                end
                if Ch < 36
                    dc_data = squeeze(dc_nocorrection(stim_start:stim_start+512-1,1,Ch)); 
                    Real_HbO(end+1,:)       =   dc_data; 
                    % save the processed data
                    HbO_Spline(end+1,:)     =   squeeze(dc_Spline(stim_start:stim_start+512-1,1,Ch));
                    HbO_Wavelet05(end+1,:)  =   squeeze(dc_Wavelet05(stim_start:stim_start+512-1,1,Ch));
                    HbO_Wavelet35(end+1,:)  =   squeeze(dc_Wavelet35(stim_start:stim_start+512-1,1,Ch));
                    HbO_Kalman(end+1,:)     =   squeeze(dc_Kalman(stim_start:stim_start+512-1,1,Ch));
                    HbO_PCA99(end+1,:)      =   squeeze(dc_PCA99(stim_start:stim_start+512-1,1,Ch));
                    HbO_PCA50(end+1,:)      =   squeeze(dc_PCA50(stim_start:stim_start+512-1,1,Ch));
                    HbO_Cbsi(end+1,:)       =   squeeze(dc_Cbsi(stim_start:stim_start+512-1,1,Ch));
                    % save the number of trials left
                    [n_MA_Spline,~,~]       =   CalMotionArtifact(tIncChAuto_after_Spline(stim_start:stim_start+512-1,Ch));
                    n_Spline_HbO            =   n_Spline_HbO + n_MA_Spline;
                    [n_MA_Wavelet05,~,~]    =   CalMotionArtifact(tIncAuto_Wavelet05(stim_start:stim_start+512-1,Ch));
                    n_Wavelet05_HbO         =   n_Wavelet05_HbO + n_MA_Wavelet05;
                    [n_MA_Wavelet35,~,~]    =   CalMotionArtifact(tIncAuto_Wavelet35(stim_start:stim_start+512-1,Ch));
                    n_Wavelet35_HbO         =   n_Wavelet35_HbO + n_MA_Wavelet35;
                    [n_MA_Kalman,~,~]       =   CalMotionArtifact(tIncAuto_Kalman(stim_start:stim_start+512-1,Ch));
                    n_Kalman_HbO            =   n_Kalman_HbO + n_MA_Kalman;
                    [n_MA_PCA99,~,~]        =   CalMotionArtifact(tIncAuto_PCA99(stim_start:stim_start+512-1,Ch));
                    n_PCA99_HbO             =   n_PCA99_HbO + n_MA_PCA99;
                    [n_MA_PCA50,~,~]        =   CalMotionArtifact(tIncAuto_PCA50(stim_start:stim_start+512-1,Ch));
                    n_PCA50_HbO             =   n_PCA50_HbO + n_MA_PCA50;
                    [n_MA_Cbsi,~,~]         =   CalMotionArtifact(tIncAuto_Cbsi(stim_start:stim_start+512-1,Ch));
                    n_Cbsi_HbO              =   n_Cbsi_HbO + n_MA_Cbsi;
                else
                    dc_data = squeeze(dc(stim_start:stim_start+512-1,2,Ch-36)); 
                    Real_HbR(end+1,:)       =   dc_data; 
                    
                    HbR_Spline(end+1,:)     =   squeeze(dc_Spline(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Wavelet05(end+1,:)    =   squeeze(dc_Wavelet05(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Wavelet35(end+1,:)    =   squeeze(dc_Wavelet35(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Kalman(end+1,:)     =   squeeze(dc_Kalman(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_PCA50(end+1,:)      =   squeeze(dc_PCA50(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_PCA99(end+1,:)      =   squeeze(dc_PCA99(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Cbsi(end+1,:)       =   squeeze(dc_Cbsi(stim_start:stim_start+512-1,2,Ch-36));
                    % save the number of trials left
                    [n_MA_Spline,~,~]       =   CalMotionArtifact(tIncChAuto_after_Spline(stim_start:stim_start+512-1,Ch));
                    n_Spline_HbR            =   n_Spline_HbR + n_MA_Spline;
                    [n_MA_Wavelet05,~,~]      =   CalMotionArtifact(tIncAuto_Wavelet05(stim_start:stim_start+512-1,Ch));
                    n_Wavelet05_HbR           =   n_Wavelet05_HbR + n_MA_Wavelet05;
                    [n_MA_Wavelet35,~,~]      =   CalMotionArtifact(tIncAuto_Wavelet35(stim_start:stim_start+512-1,Ch));
                    n_Wavelet35_HbR           =   n_Wavelet35_HbR + n_MA_Wavelet35;
                    [n_MA_Kalman,~,~]       =   CalMotionArtifact(tIncAuto_Kalman(stim_start:stim_start+512-1,Ch));
                    n_Kalman_HbR            =   n_Kalman_HbR + n_MA_Kalman;
                    [n_MA_PCA99,~,~]        =   CalMotionArtifact(tIncAuto_PCA99(stim_start:stim_start+512-1,Ch));
                    n_PCA99_HbR             =   n_PCA99_HbR + n_MA_PCA99;
                    [n_MA_PCA50,~,~]        =   CalMotionArtifact(tIncAuto_PCA50(stim_start:stim_start+512-1,Ch));
                    n_PCA50_HbR             =   n_PCA50_HbR + n_MA_PCA50;
                    [n_MA_Cbsi,~,~]         =   CalMotionArtifact(tIncAuto_Cbsi(stim_start:stim_start+512-1,Ch));
                    n_Cbsi_HbR              =   n_Cbsi_HbR + n_MA_Cbsi;
                end
            end
        end
    end
end


%%
n_MA_total_HbO = sum(n_MA_HbO_list);
n_MA_total_HbR = sum(n_MA_HbR_list);
MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet35_HbO,n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet35_HbR,n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO];
%% save
save('Processed_data/Real_HbO.mat','Real_HbO')
save('Processed_data/HbO_Spline.mat','HbO_Spline')
save('Processed_data/HbO_Wavelet05.mat','HbO_Wavelet05')
save('Processed_data/HbO_Wavelet35.mat','HbO_Wavelet35')
save('Processed_data/HbO_Kalman.mat','HbO_Kalman')
save('Processed_data/HbO_PCA99.mat','HbO_PCA99')
save('Processed_data/HbO_PCA50.mat','HbO_PCA50')
save('Processed_data/HbO_Cbsi.mat','HbO_Cbsi')

save('Processed_data/Real_HbR.mat','Real_HbR')
save('Processed_data/HbR_Spline.mat','HbR_Spline')
save('Processed_data/HbR_Wavelet05.mat','HbR_Wavelet05')
save('Processed_data/HbR_Wavelet35.mat','HbR_Wavelet35')
save('Processed_data/HbR_Kalman.mat','HbR_Kalman')
save('Processed_data/HbR_PCA99.mat','HbR_PCA99')
save('Processed_data/HbR_PCA50.mat','HbR_PCA50')
save('Processed_data/HbR_Cbsi.mat','HbR_Cbsi')

save('Processed_data/MA_list.mat','MA_list')
