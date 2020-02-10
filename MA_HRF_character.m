% Extract qualified data: 0~25s with some Motion artifacts.
% Save the data as Testing data.
% Process data via PCA etc. and save as output HRF and n_MA_left
% Extract motion artifact peak distribution and HRF peak values
% distribution
% what does motion artifact look like in the real data?
% function [pd_HbO,pd_HbR] = MA_HRF_character()

clear all
clc
close all
%% add homer path
pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define variables to be calulate
diff_HbO_list = [];
diff_HbR_list = [];
n_MA_HbO_list = [];
n_MA_HbR_list = [];
HRF_HbO_list = [];
HRF_HbR_list = [];
Real_HbO    = [];
Real_HbR    = [];
HbO_Spline  = [];
HbR_Spline  = [];
HbO_Wavelet = [];
HbR_Wavelet = [];
HbO_Kalman  = [];
HbR_Kalman  = [];
HbO_PCA97   = [];
HbR_PCA97   = [];
HbO_PCA80   = [];
HbR_PCA80   = [];
HbO_Cbsi    = [];
HbR_Cbsi    = [];

n_Spline_HbO = 0;
n_Spline_HbR = 0;
n_Wavelet_HbO = 0;
n_Wavelet_HbR = 0;
n_Kalman_HbO = 0;
n_Kalman_HbR = 0;
n_PCA97_HbO = 0;
n_PCA97_HbR = 0;
n_PCA80_HbO = 0;
n_PCA80_HbR = 0;
n_Cbsi_HbO = 0;
n_Cbsi_HbR = 0;

STD = 10;
%% load fNIRS file
% loop subfolders and files
DataDir = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2';
Subfolders = dir(DataDir);
for folder = 4:length(Subfolders)
    name = Subfolders(folder).name;
    if strcmp(name,'L8')
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
        dc              =   hmrOD2Conc(dod,SD,[6  6]);
        %% standard processing with PCA97
        SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                     =   hmrIntensity2OD(d);
        [dod_PCA97,~,~,~,~]     =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,30,200,0.97,5);
        [~,tIncAuto_PCA97]      =   hmrMotionArtifactByChannel(dod_PCA97,t,SD,tIncMan,0.5,1,STD,200);
        dod_PCA97               =   hmrBandpassFilt(dod_PCA97,t,0,0.5);
        dc_PCA97                =   hmrOD2Conc(dod_PCA97,SD,[6  6]);
        %% standard processing with PCA80
        SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                     =   hmrIntensity2OD(d);
        [dod_PCA80,~,~,~,~]     =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,30,200,0.80,5);
        [~, tIncAuto_PCA80]     =   hmrMotionArtifactByChannel(dod_PCA80,t,SD,tIncMan,0.5,1,STD,200);
        dod_PCA80               =   hmrBandpassFilt(dod_PCA80,t,0,0.5);
        dc_PCA80                =   hmrOD2Conc(dod_PCA80,SD,[6  6]);
        %% standard processing with Spline
        SD                              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod                             =   hmrIntensity2OD(d);
        [~,tIncChAuto_before_Spline]    =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
        [dod_Spline]                    =   hmrMotionCorrectSpline(dod,t,SD,tIncChAuto_before_Spline,0.99);
        [~,tIncChAuto_after_Spline]     =   hmrMotionArtifactByChannel(dod_Spline,t,SD,tIncMan,0.5,1,STD,200);
        dod_Spline                      =   hmrBandpassFilt(dod_Spline,t,0,0.5);
        dc_Spline                       =   hmrOD2Conc(dod_Spline,SD,[6  6]);
        %% standard processing with Wavelet
        SD                  =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0); 
        dod                 =   hmrIntensity2OD(d);
        [dod_Wavelet]       =   hmrMotionCorrectWavelet(dod,SD,0.1);
        [~,tIncAuto_Wavelet]=   hmrMotionArtifactByChannel(dod_Wavelet,t,SD,tIncMan,0.5,1,STD,200);
        dod_Wavelet         =   hmrBandpassFilt(dod_Wavelet,t,0,0.5);
        dc_Wavelet          =   hmrOD2Conc(dod_Wavelet,SD,[6  6]);
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
                % count how many motion artifacts within one trial and peak
                % height 
                if Ch < 36
                    n_MA_HbO_list(end+1) = n_MA;
                else
                    n_MA_HbR_list(end+1) = n_MA;
                end
                for i = 1:n_MA
                    start_point = runstarts(i)-1;
                    end_point   = runends(i)-1;
                    if Ch < 36
                        MA_HbO      = squeeze(dc(stim_start+start_point:stim_start+end_point,1,Ch));
                        diff_HbO    = max(MA_HbO)-min(MA_HbO);
                        diff_HbO_list(end+1) = diff_HbO;

                    else
                        MA_HbR      = squeeze(dc(stim_start+start_point:stim_start+end_point,2,Ch-36));
                        diff_HbR    = max(MA_HbR)-min(MA_HbR);
                        diff_HbR_list(end+1) = diff_HbR;
                    end
                end
                if Ch < 36
                    % save the 0-65s data
                    dc_data = squeeze(dc(stim_start:stim_start+512-1,1,Ch)); 
                    Real_HbO(end+1,:)       =   dc_data; 
                    tInc                    =   tIncChAuto(stim_start:stim_start+512-1,Ch);
                    Real_HbO_wo_MA          =   dc_data(find(tInc==1)); 
                    HRF_HbO_list(end+1)     =   max(Real_HbO_wo_MA)-min(Real_HbO_wo_MA);
                    % save the processed data
                    HbO_Spline(end+1,:)     =   squeeze(dc_Spline(stim_start:stim_start+512-1,1,Ch));
                    HbO_Wavelet(end+1,:)    =   squeeze(dc_Wavelet(stim_start:stim_start+512-1,1,Ch));
                    HbO_Kalman(end+1,:)     =   squeeze(dc_Kalman(stim_start:stim_start+512-1,1,Ch));
                    HbO_PCA97(end+1,:)      =   squeeze(dc_PCA97(stim_start:stim_start+512-1,1,Ch));
                    HbO_PCA80(end+1,:)      =   squeeze(dc_PCA80(stim_start:stim_start+512-1,1,Ch));
                    HbO_Cbsi(end+1,:)       =   squeeze(dc_Cbsi(stim_start:stim_start+512-1,1,Ch));
                    % save the number of trials left
                    [n_MA_Spline,~,~]       =   CalMotionArtifact(tIncChAuto_after_Spline(stim_start:stim_start+512-1,Ch));
                    n_Spline_HbO            =   n_Spline_HbO + n_MA_Spline;
                    [n_MA_Wavelet,~,~]      =   CalMotionArtifact(tIncAuto_Wavelet(stim_start:stim_start+512-1,Ch));
                    n_Wavelet_HbO           =   n_Spline_HbO + n_MA_Wavelet;
                    [n_MA_Kalman,~,~]       =   CalMotionArtifact(tIncAuto_Kalman(stim_start:stim_start+512-1,Ch));
                    n_Kalman_HbO            =   n_Spline_HbO + n_MA_Kalman;
                    [n_MA_PCA97,~,~]        =   CalMotionArtifact(tIncAuto_PCA97(stim_start:stim_start+512-1,Ch));
                    n_PCA97_HbO             =   n_PCA97_HbO + n_MA_PCA97;
                    [n_MA_PCA80,~,~]        =   CalMotionArtifact(tIncAuto_PCA80(stim_start:stim_start+512-1,Ch));
                    n_PCA80_HbO             =   n_PCA80_HbO + n_MA_PCA80;
                    [n_MA_Cbsi,~,~]         =   CalMotionArtifact(tIncAuto_Cbsi(stim_start:stim_start+512-1,Ch));
                    n_Cbsi_HbO              =   n_Cbsi_HbO + n_MA_Cbsi;
                else
                    dc_data                 =   squeeze(dc(stim_start:stim_start+512-1,2,Ch-36));
                    Real_HbR(end+1,:)       =   dc_data; 
                    tInc                    =   tIncChAuto(stim_start:stim_start+512-1,Ch);
                    Real_HbR_wo_MA          =   dc_data(find(tInc==1)); 
                    HRF_HbR_list(end+1)     =   max(Real_HbR_wo_MA)-min(Real_HbR_wo_MA);
                    HbR_Spline(end+1,:)     =   squeeze(dc_Spline(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Wavelet(end+1,:)    =   squeeze(dc_Wavelet(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Kalman(end+1,:)     =   squeeze(dc_Kalman(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_PCA80(end+1,:)      =   squeeze(dc_PCA80(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_PCA97(end+1,:)      =   squeeze(dc_PCA97(stim_start:stim_start+512-1,2,Ch-36));
                    HbR_Cbsi(end+1,:)       =   squeeze(dc_Cbsi(stim_start:stim_start+512-1,2,Ch-36));
                    % save the number of trials left
                    [n_MA_Spline,~,~]       =   CalMotionArtifact(tIncChAuto_after_Spline(stim_start:stim_start+512-1,Ch));
                    n_Spline_HbR            =   n_Spline_HbR + n_MA_Spline;
                    [n_MA_Wavelet,~,~]      =   CalMotionArtifact(tIncAuto_Wavelet(stim_start:stim_start+512-1,Ch));
                    n_Wavelet_HbR           =   n_Spline_HbR + n_MA_Wavelet;
                    [n_MA_Kalman,~,~]       =   CalMotionArtifact(tIncAuto_Kalman(stim_start:stim_start+512-1,Ch));
                    n_Kalman_HbR            =   n_Spline_HbR + n_MA_Kalman;
                    [n_MA_PCA97,~,~]        =   CalMotionArtifact(tIncAuto_PCA97(stim_start:stim_start+512-1,Ch));
                    n_PCA97_HbR             =   n_PCA97_HbR + n_MA_PCA97;
                    [n_MA_PCA80,~,~]        =   CalMotionArtifact(tIncAuto_PCA80(stim_start:stim_start+512-1,Ch));
                    n_PCA80_HbR             =   n_PCA80_HbR + n_MA_PCA80;
                    [n_MA_Cbsi,~,~]         =   CalMotionArtifact(tIncAuto_Cbsi(stim_start:stim_start+512-1,Ch));
                    n_Cbsi_HbR              =   n_Cbsi_HbR + n_MA_Cbsi;
                end
                    % and the HRF height without motion artifact
            end
        end
    end
end
%%
pd_diff_HbO = fitdist(diff_HbO_list','gamma');
pd_diff_HbR = fitdist(diff_HbR_list','gamma');
pd_n_HbO = fitdist(n_MA_HbO_list','gamma');
pd_n_HbR = fitdist(n_MA_HbR_list','gamma');
pd_HRF_HbO = fitdist(HRF_HbO_list','gamma');
pd_HRF_HbR = fitdist(HRF_HbR_list','gamma');
save('Processed_data/pds.mat','pd_diff_HbO','pd_diff_HbR','pd_n_HbO','pd_n_HbR','pd_HRF_HbO','pd_HRF_HbR');
figure
histogram(diff_HbO_list);hold on;
% simu = gamrnd(pd_HbO.a,pd_HbO.b,1500,1);
title('dist. of peak/shift distance in HbO')
savefig('Figures/dist. of peakshift distance in HbO.fig')

figure
histogram(diff_HbR_list);hold on;
title('dist. of peak/shift distance in HbR')
savefig('Figures/dist. of peakshift distance in HbR.fig')

figure
histogram(n_MA_HbO_list);hold on;
title('dist. of No. of motion artifact for HbO')
savefig('Figures/dist. of No. of motion artifact for HbO.fig')

figure
histogram(n_MA_HbR_list);hold on;
title('dist. of No. of motion artifact for HbR')
savefig('Figures/dist. of No. of motion artifact for HbR.fig')

figure
histogram(HRF_HbO_list);
title('dist. of HRF change in HbO')
savefig('Figures/dist. of HRF change in HbO.fig')

figure
histogram(HRF_HbR_list);
title('dist. of HRF change in HbR')
savefig('Figures/dist. of HRF change in HbO.fig')


%%
n_MA_total_HbO = sum(n_MA_HbO_list);
n_MA_total_HbR = sum(n_MA_HbR_list);
MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet_HbO,n_Kalman_HbO,n_PCA97_HbO,n_PCA80_HbO,n_Cbsi_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet_HbR,n_Kalman_HbR,n_PCA97_HbR,n_PCA80_HbR,n_Cbsi_HbO];
%% save
save('Processed_data/Real_HbO.mat','Real_HbO')
save('Processed_data/HbO_Spline.mat','HbO_Spline')
save('Processed_data/HbO_Wavelet.mat','HbO_Wavelet')
save('Processed_data/HbO_Kalman.mat','HbO_Kalman')
save('Processed_data/HbO_PCA97.mat','HbO_PCA97')
save('Processed_data/HbO_PCA80.mat','HbO_PCA80')
save('Processed_data/HbO_Cbsi.mat','HbO_Cbsi')

save('Processed_data/Real_HbR.mat','Real_HbR')
save('Processed_data/HbR_Spline.mat','HbR_Spline')
save('Processed_data/HbR_Wavelet.mat','HbR_Wavelet')
save('Processed_data/HbR_Kalman.mat','HbR_Kalman')
save('Processed_data/HbR_PCA97.mat','HbR_PCA97')
save('Processed_data/HbR_PCA80.mat','HbR_PCA80')
save('Processed_data/HbR_Cbsi.mat','HbR_Cbsi')

save('Processed_data/MA_list.mat','MA_list')
