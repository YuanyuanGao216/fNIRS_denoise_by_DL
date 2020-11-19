%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

%% load data
clear all
load('Processed_data/SimulateData.mat','HRF_test_noised')

[m,n] = size(HRF_test_noised);
HbO_noised = HRF_test_noised(1:m/2,:);
HbR_noised = HRF_test_noised(m/2+1:end,:);
%% define variables to be calulate
Proc_data
HbO_Spline  = [];
HbR_Spline  = [];
HbO_Wavelet01 = [];
HbR_Wavelet01 = [];
HbO_Kalman  = [];
HbR_Kalman  = [];
HbO_PCA99   = [];
HbR_PCA99   = [];
HbO_PCA79   = [];
HbR_PCA79   = [];
HbO_Cbsi    = [];
HbR_Cbsi    = [];

n_Spline_HbO = 0;
n_Spline_HbR = 0;
n_Wavelet01_HbO = 0;
n_Wavelet01_HbR = 0;
n_Kalman_HbO = 0;
n_Kalman_HbR = 0;
n_PCA99_HbO = 0;
n_PCA99_HbR = 0;
n_PCA79_HbO = 0;
n_PCA79_HbR = 0;
n_Cbsi_HbO = 0;
n_Cbsi_HbR = 0;

global fs_new
fs_new = 7;
global rt
global pt
rt =   40; %resting time 5.7s
pt =   512-40; %performance time 65.536s - 5.7s
STD = 10;

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
% %% Cbsi
% T_Cbsi = 0;
% 
% for i = 1:m/2
%     dc_HbO  =   HbO_noised(i,:);
%     dc_HbR  =   HbR_noised(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
% 
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     dod     =   hmrBandpassFilt(dod,t,0,0.5);
%     dc      =   hmrOD2Conc(dod,SD1,[6  6]);
%     tic
%     [dc_Cbsi] = hmrMotionCorrectCbsi(dc,SD1,0);
%     T_Cbsi = T_Cbsi + toc;
%     [dc_avg_Cbsi, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_Cbsi, s', t, [-39/fs_new (512-40)/fs_new] ); 
%     
%     dc_predict_HbO = squeeze(dc_avg_Cbsi(:,1));
%     dc_predict_HbR = squeeze(dc_avg_Cbsi(:,2));
%     dod_Cbsi = hmrConc2OD(dc_Cbsi,SD1,[6  6]);
%     [~,tIncAuto_Cbsi] = hmrMotionArtifactByChannel(dod_Cbsi,t,SD1,tIncMan,0.5,1,STD,200);
%     
%     HbO_Cbsi(end+1,:) = dc_predict_HbO';
%     HbR_Cbsi(end+1,:) = dc_predict_HbR';
%     
%     [n_MA_HbO_Cbsi,~,~]    =   CalMotionArtifact(tIncAuto_Cbsi(:,1));
%     [n_MA_HbR_Cbsi,~,~]    =   CalMotionArtifact(tIncAuto_Cbsi(:,2));
%     n_Cbsi_HbO = n_Cbsi_HbO + n_MA_HbO_Cbsi;
%     n_Cbsi_HbR = n_Cbsi_HbR + n_MA_HbR_Cbsi;
% end
% save('Processed_data/Testing_Cbsi.mat','HbO_Cbsi','HbR_Cbsi','n_Cbsi_HbO','n_Cbsi_HbR')

% %% Spline 99
% p   =   0.99;
% T_Spline = 0;
% 
% for i = 1:m/2
%     dc_HbO  =   HbO_noised(i,:);
%     dc_HbR  =   HbR_noised(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
% 
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     tic
%     [~,tIncChAuto_before_Spline]    =   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,200);
%     [dod_Spline]                    =   hmrMotionCorrectSpline(dod,t,SD1,tIncChAuto_before_Spline,p);
%     T_Spline                        =   T_Spline + toc;
%     [~,tIncAuto_Spline]             =   hmrMotionArtifactByChannel(dod_Spline,t,SD1,tIncMan,0.5,1,STD,200);
%     dod_Spline                      =   hmrBandpassFilt(dod_Spline,t,0,0.5);
%     dc_Spline                       =   hmrOD2Conc(dod_Spline,SD1,[6  6]);
%     [dc_avg_Spline, ~, ~, ~, ~, ~]  = hmrBlockAvg(dc_Spline, s', t, [-39/fs_new (512-40)/fs_new] ); 
% 
%     dc_predict_HbO                  =   squeeze(dc_avg_Spline(:,1));
%     dc_predict_HbR                  =   squeeze(dc_avg_Spline(:,2));
%     HbO_Spline(end+1,:)             =   dc_predict_HbO';
%     HbR_Spline(end+1,:)             =   dc_predict_HbR';
%     
%     [n_MA_HbO_Spline,~,~]           =   CalMotionArtifact(tIncAuto_Spline(:,1));
%     [n_MA_HbR_Spline,~,~]           =   CalMotionArtifact(tIncAuto_Spline(:,2));
%     n_Spline_HbO                    =   n_Spline_HbO + n_MA_HbO_Spline;
%     n_Spline_HbR                    =   n_Spline_HbR + n_MA_HbR_Spline;
% end
% save('Processed_data/Testing_Spline.mat','HbO_Spline','HbR_Spline','n_Spline_HbO','n_Spline_HbR')
%% PCA79
sigma   =  0.79;
T_PCA79 =   0;

for i = 1:m/2
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';

    dod     =   hmrConc2OD( dc, SD1, ppf );
    tic
    [dod_PCA,~,~,~,~]   =   hmrMotionCorrectPCArecurse(dod,t,SD1,tIncMan,0.5,1,STD,200,sigma,5);
    T_PCA79 = T_PCA79 + toc;
    [~,tIncAuto_PCA]    =   hmrMotionArtifactByChannel(dod_PCA,t,SD1,tIncMan,0.5,1,STD,200);
    dod_PCA             =   hmrBandpassFilt(dod_PCA,t,0,0.5);
    dc_PCA              =   hmrOD2Conc(dod_PCA,SD1,[6  6]);
    [dc_avg_PCA79, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_PCA, s', t, [-39/fs_new (512-40)/fs_new] ); 

    dc_predict_HbO = squeeze(dc_avg_PCA79(:,1));
    dc_predict_HbR = squeeze(dc_avg_PCA79(:,2));
    dc_predict = [dc_predict_HbO',dc_predict_HbR'];
    HbO_PCA79(end+1,:) = dc_predict_HbO';
    HbR_PCA79(end+1,:) = dc_predict_HbR';
    
    [n_MA_HbO_PCA,~,~]    =   CalMotionArtifact(tIncAuto_PCA(:,1));
    [n_MA_HbR_PCA,~,~]    =   CalMotionArtifact(tIncAuto_PCA(:,2));
    n_PCA79_HbO = n_PCA79_HbO + n_MA_HbO_PCA;
    n_PCA79_HbR = n_PCA79_HbR + n_MA_HbR_PCA;
end
save('Processed_data/Testing_PCA79.mat','HbO_PCA79','HbR_PCA79','n_PCA79_HbO','n_PCA79_HbR')


% %% PCA99
% T_PCA99 = 0;
% sigma   =   0.99;
% 
% for i = 1:m/2
%     dc_HbO  =   HbO_noised(i,:);
%     dc_HbR  =   HbR_noised(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
% 
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     tic
%     [dod_PCA,~,~,~,~]   =   hmrMotionCorrectPCArecurse(dod,t,SD1,tIncMan,0.5,1,STD,200,sigma,5);
%     T_PCA99 = T_PCA99 + toc;
%     [~,tIncAuto_PCA]    =   hmrMotionArtifactByChannel(dod_PCA,t,SD1,tIncMan,0.5,1,STD,200);
%     dod_PCA             =   hmrBandpassFilt(dod_PCA,t,0,0.5);
%     dc_PCA              =   hmrOD2Conc(dod_PCA,SD1,[6  6]);
%     [dc_avg_PCA99, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_PCA, s', t, [-39/fs_new (512-40)/fs_new] ); 
%     
%     dc_predict_HbO = squeeze(dc_avg_PCA99(:,1));
%     dc_predict_HbR = squeeze(dc_avg_PCA99(:,2));
%     dc_predict = [dc_predict_HbO',dc_predict_HbR'];
%     HbO_PCA99(end+1,:) = dc_predict_HbO';
%     HbR_PCA99(end+1,:) = dc_predict_HbR';
%     
%     [n_MA_HbO_PCA,~,~]    =   CalMotionArtifact(tIncAuto_PCA(:,1));
%     [n_MA_HbR_PCA,~,~]    =   CalMotionArtifact(tIncAuto_PCA(:,2));
%     n_PCA99_HbO = n_PCA99_HbO + n_MA_HbO_PCA;
%     n_PCA99_HbR = n_PCA99_HbR + n_MA_HbR_PCA;
% end
% save('Processed_data/Testing_PCA99.mat','HbO_PCA99','HbR_PCA99','n_PCA99_HbO','n_PCA99_HbR')

% %% Kalman
% T_Kalman = 0;
% 
% for i = 1:m/2
%     dc_HbO  =   HbO_noised(i,:);
%     dc_HbR  =   HbR_noised(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
% 
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     y               =   dod;
%     oscFreq         =   [0,0.01,0.001,0.0001];
%     xo              =   ones(1,length(oscFreq)+1)*y(1,1);
%     Po              =   ones(1,length(oscFreq)+1)*(y(1,1)^2);
%     Qo              =   zeros(1,length(oscFreq)+1);
%     hrfParam        =   [2 2];
%     tic
%     [x, yStim,dod_Kalman,y,C,Q] = hmrKalman2( y, s, t, xo, Po, Qo, 'box', hrfParam, oscFreq );
%     T_Kalman = T_Kalman + toc;
%     [~,tIncAuto_Kalman]   =   hmrMotionArtifactByChannel(dod_Kalman,t,SD1,tIncMan,0.5,1,STD,200);
%     dod_Kalman          =   hmrBandpassFilt(dod_Kalman,t,0,0.5);
%     dc_Kalman           =   hmrOD2Conc(dod_Kalman,SD1,[6  6]);
%     [dc_avg_Kalman, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_Kalman, s', t, [-39/fs_new (512-40)/fs_new] ); 
% 
%     dc_predict_HbO = squeeze(dc_avg_Kalman(:,1));
%     dc_predict_HbR = squeeze(dc_avg_Kalman(:,2));
%     
%     HbO_Kalman(end+1,:) = dc_predict_HbO';
%     HbR_Kalman(end+1,:) = dc_predict_HbR';
%     
%     [n_MA_HbO_Kalman,~,~]    =   CalMotionArtifact(tIncAuto_Kalman(:,1));
%     [n_MA_HbR_Kalman,~,~]    =   CalMotionArtifact(tIncAuto_Kalman(:,2));
%     n_Kalman_HbO = n_Kalman_HbO + n_MA_HbO_Kalman;
%     n_Kalman_HbR = n_Kalman_HbR + n_MA_HbR_Kalman;
% end
% save('Processed_data/Testing_Kalman.mat','HbO_Kalman','HbR_Kalman','n_Kalman_HbO','n_Kalman_HbR')

% %% Wavelet01
% T_Wavelet01 =   0;
% alpha       =   0.1;
% 
% for i = 1:m/2
%     dc_HbO  =   HbO_noised(i,:);
%     dc_HbR  =   HbR_noised(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
% 
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     tic
%     [dod_Wavelet01]       =   hmrMotionCorrectWavelet(dod,SD1,alpha);
%     T_Wavelet01 = T_Wavelet01 + toc;
%     [~,tIncAuto_Wavelet01]=   hmrMotionArtifactByChannel(dod_Wavelet01,t,SD1,tIncMan,0.5,1,STD,200);
%     dod_Wavelet01         =   hmrBandpassFilt(dod_Wavelet01,t,0,0.5);
%     dc_Wavelet01          =   hmrOD2Conc(dod_Wavelet01,SD1,[6  6]);
%     [dc_avg_Wavelet01, ~, ~, ~, ~, ~] = hmrBlockAvg(dc_Wavelet01, s', t, [-39/fs_new (512-40)/fs_new] ); 
%     
%     dc_predict_HbO = squeeze(dc_avg_Wavelet01(:,1));
%     dc_predict_HbR = squeeze(dc_avg_Wavelet01(:,2));
%     
%     HbO_Wavelet01(end+1,:) = dc_predict_HbO';
%     HbR_Wavelet01(end+1,:) = dc_predict_HbR';
%     
%     [n_MA_HbO_Wavelet01,~,~]    =   CalMotionArtifact(tIncAuto_Wavelet01(:,1));
%     [n_MA_HbR_Wavelet01,~,~]    =   CalMotionArtifact(tIncAuto_Wavelet01(:,2));
%     n_Wavelet01_HbO = n_Wavelet01_HbO + n_MA_HbO_Wavelet01;
%     n_Wavelet01_HbR = n_Wavelet01_HbR + n_MA_HbR_Wavelet01;
% end
% save('Processed_data/Testing_Wavelet01.mat','HbO_Wavelet01','HbR_Wavelet01','n_Wavelet01_HbO','n_Wavelet01_HbR')


%% NN count noise
% filepath = 'Processed_data/Test_NN_8layers.mat';
% 
% load(filepath)% File exists.
% Hb_NN = Y_test_predict;
% 
% HbO_NN = Hb_NN(:,1:512);
% HbR_NN = Hb_NN(:,512+1:end);
% HbO_NN = reshape(HbO_NN',[512*5,644]);
% HbR_NN = reshape(HbR_NN',[512*5,644]);
% HbO_NN = HbO_NN';
% HbR_NN = HbR_NN';
% n_NN_HbO = 0;
% n_NN_HbR = 0;
% for i = 1:m/2
%     dc_HbO  =   HbO_NN(i,:);
%     dc_HbR  =   HbR_NN(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     [~,tIncAuto]=   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,200);
%     
%     [n_MA_HbO,~,~]    =   CalMotionArtifact(tIncAuto(:,1));
%     [n_MA_HbR,~,~]    =   CalMotionArtifact(tIncAuto(:,2));
%     n_NN_HbO = n_NN_HbO + n_MA_HbO;
%     n_NN_HbR = n_NN_HbR + n_MA_HbR;
% end
% save('Processed_data/Testing_NN.mat','n_NN_HbO','n_NN_HbR')
%% output time
time_file = fopen('Processed_data/time.txt','a+');
fprintf(time_file,'Time for Spline is %f\n',T_Spline);
fprintf(time_file,'Time for Wavelet01 is %f\n',T_Wavelet01);
fprintf(time_file,'Time for Kalman is %f\n',T_Kalman);
fprintf(time_file,'Time for PCA99 is %f\n',T_PCA99);
fprintf(time_file,'Time for PCA79 is %f\n',T_PCA79);
fprintf(time_file,'Time for Cbsi is %f\n',T_Cbsi);
% % T_Spline = 8.637879;
% % T_Wavelet05 = 1111.117784;
% % T_Wavelet35 = 1102.784217;
% % T_Kalman = 55.251534;
% % T_PCA99 = 6.714928;
% % T_PCA50 = 7.102599;
% % T_Cbsi = 0.461004;
% % T_DAE = 0.3288;
% % T_DAE = 0.3555;for 4 layers
% fprintf('Time for DAE is %f\n',T_DAE)
% 
T_list = [T_Spline,T_Wavelet01,T_Kalman,T_PCA99,T_PCA79,T_Cbsi,T_DAE];
labels = {'Spline','Wavelet01','Kalman','PCA99','PCA79','Cbsi','DAE'};
figure
b = bar(T_list,'facecolor',[108, 171, 215]./256,'edgecolor',[108, 171, 215]./256);

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints+30;
ydata = b(1).YData;
labels1 = strings(1,8);
for i = 1:size(ydata,2)
    string_data = sprintf('%.1f',ydata(i));
    labels1(i) = string_data;
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

ylabel('Computation time (s)')
title('Computation time on testing dataset')
set(gca, 'XTick', 1:size(T_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
set(gcf,'Position',[218   460   330   215]);
ylim([0 1250])
xlim([0.5 8.5])
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [200, 14, 80]./255;
b.FaceColor = 'flat';
b.CData(9,:) = [200, 14, 80]./255;
box off