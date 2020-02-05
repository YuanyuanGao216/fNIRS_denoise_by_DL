%Plot Motion artifact removal results on experimental data by conventional methods
function Plot_MAR_results()
clear all
clc
close all
%% add homer path
pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';;
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% load fNIRS file
folder = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2';
file = 'NIRS-2019-08-10_005.nirs';
fNIRS_data = load([folder,'/',file],'-mat');
%% define variables
d           =   fNIRS_data.d;
SD          =   fNIRS_data.SD;
t           =   fNIRS_data.t;
tIncMan     =   ones(size(t));
s           =   fNIRS_data.s;
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

[tIncAuto_Kalman]   =   hmrMotionArtifact(dod_Kalman,t,SD,tIncMan,0.5,1,30,200);
dod_Kalman          =   hmrBandpassFilt(dod_Kalman,t,0,0.5);
dc_Kalman           =   hmrOD2Conc(dod_Kalman,SD,[6  6]);

%% standard processing with Kalman filtering
% SD                  =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
% 
% dod                 =   hmrIntensity2OD(d);
% 
% [dod_Kalman]        =   Kalman_dod(dod);
% 
% [tIncAuto_Kalman]   =   hmrMotionArtifact(dod_Kalman,t,SD,tIncMan,0.5,1,30,200);
% 
% dod_Kalman          =   hmrBandpassFilt(dod_Kalman,t,0,0.5);
% 
% dc_Kalman           =   hmrOD2Conc(dod_Kalman,SD,[6  6]);
%% standard processing wo correction
SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);

dod                     =   hmrIntensity2OD(d);

[tIncAuto]              =   hmrMotionArtifact(dod,t,SD,tIncMan,0.5,1,30,200);

dod                     =   hmrBandpassFilt(dod,t,0,0.5);

dc                      =   hmrOD2Conc(dod,SD,[6  6]);
%% standard processing with PCA
SD                      =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);

dod                     =   hmrIntensity2OD(d);

[dod_PCA,~,~,~,~]       =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,30,200,0.97,5);

[tIncAuto_PCA]          =   hmrMotionArtifact(dod_PCA,t,SD,tIncMan,0.5,1,30,200);

dod_PCA                 =   hmrBandpassFilt(dod_PCA,t,0,0.5);

dc_PCA                  =   hmrOD2Conc(dod_PCA,SD,[6  6]);
%% standard processing with Spline
SD                              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);

dod                             =   hmrIntensity2OD(d);

[~,tIncChAuto_before_Spline]    =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,30,200);

[dod_Spline]                    =   hmrMotionCorrectSpline(dod,t,SD,tIncChAuto_before_Spline,0.99);

[~,tIncChAuto_after_Spline]     =   hmrMotionArtifactByChannel(dod_Spline,t,SD,tIncMan,0.5,1,30,200);

dod_Spline                      =   hmrBandpassFilt(dod_Spline,t,0,0.5);

dc_Spline                       =   hmrOD2Conc(dod_Spline,SD,[6  6]);
%% standard processing with Wavelet
SD                  =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);

dod                 =   hmrIntensity2OD(d);

[dod_Wavelet]       =   hmrMotionCorrectWavelet(dod,SD,0.1);

[tIncAuto_Wavelet]  =   hmrMotionArtifact(dod_Wavelet,t,SD,tIncMan,0.5,1,30,200);

dod_Wavelet         =   hmrBandpassFilt(dod_Wavelet,t,0,0.5);

dc_Wavelet          =   hmrOD2Conc(dod_Wavelet,SD,[6  6]);

%% plot dc to compare

flag = 1;
while flag
    ch = input('which channel to show? ');
    if SD.MeasListAct(ch) == 0
        fprintf('this channel is pruned\n')
    else
        data_no_corr    =   dc(:,:,ch);
        data_PCA        =   dc_PCA(:,:,ch);
        data_Spline     =   dc_Spline(:,:,ch);
        data_Wavelet    =   dc_Wavelet(:,:,ch);
        data_Kalman     =   dc_Kalman(:,:,ch);

        figure('Renderer', 'painters', 'Position', [10 10 1200 600]);
        subplot(2,1,1);
        plot(data_no_corr(:,1),...
            'k-o','DisplayName','no correction');hold on;
        plot((tIncAuto+5)/1000000,...
            'k-o','DisplayName','motion artifact');hold on;
        plot(data_PCA(:,1),...
            'b-','DisplayName','PCA');          hold on;
        plot((tIncAuto_PCA+6)/1000000,...
            'b-','DisplayName','motion artifact');hold on;
        plot(data_Spline(:,1),...
            'r-','DisplayName','Spline');          hold on;
        plot((tIncChAuto_before_Spline(:,ch)+7)/1000000,...
            'r-','DisplayName','motion artifact before spline');hold on;
        plot((tIncChAuto_after_Spline(:,ch)+8)/1000000,...
            'r-','DisplayName','motion artifact after spline');hold on;
        plot(data_Wavelet(:,1),...
            'g-','DisplayName','Wavelet');          hold on;
        plot((tIncAuto_Wavelet+9)/1000000,...
            'g-','DisplayName','Wavelet');hold on;
        plot(data_Kalman(:,1),...
            'm-','DisplayName','Kalman');          hold on;
        plot((tIncAuto_Kalman+10)/1000000,...
            'm-','DisplayName','Kalman');hold on;

        title('HbO')
        hold off;
        legend show;
        legend('location','southeastoutside')

        subplot(2,1,2);
        plot(data_no_corr(:,2), 'k-o','DisplayName','no correction');hold on;
        plot((tIncAuto+5)/1000000,  'k-o','DisplayName','motion artifact');hold on;
        plot(data_PCA(:,2),     'b-','DisplayName','PCA');          hold on;
        plot((tIncAuto_PCA+6)/1000000,  'b-','DisplayName','motion artifact');hold on;
        plot(data_Spline(:,2),...
            'r-','DisplayName','Spline');          hold on;
        plot((tIncChAuto_before_Spline(:,ch+32)+7)/1000000,...
            'r-','DisplayName','before spline');hold on;
        plot((tIncChAuto_after_Spline(:,ch+32)+8)/1000000,...
            'r-','DisplayName','after spline');hold on;
        plot(data_Wavelet(:,2),...
            'g-','DisplayName','Wavelet');          hold on;
        plot((tIncAuto_Wavelet+9)/1000000,...
            'g-','DisplayName','Wavelet');hold on;
        plot(data_Kalman(:,2),...
            'm-','DisplayName','Kalman');           hold on;
        plot((tIncAuto_Kalman+10)/1000000,...
            'm-','DisplayName','Kalman');           hold on;

        title('HbR')
        legend show;
        legend('location','southeastoutside')
    end
    flag = input('Continue?');
end