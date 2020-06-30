%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

%% load data
load('Processed_data/HRF_test.mat','HRF_test')
load('Processed_data/HRF_test_noised.mat','HRF_test_noised')
[m,n] = size(HRF_test_noised);
HbO_noised = HRF_test_noised(1:m/2,:);
HbR_noised = HRF_test_noised(m/2+1:end,:);
HbO = HRF_test(1:m/2,:);
HbR = HRF_test(m/2+1:end,:);
STD = 10;
SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];
fs = 7.8125;
t  = 1/fs:1/fs:512/fs;
STD = 10;
tIncMan=ones(size(t))';
N = 100;
%% Cbsi
fprintf('cbsi\n')
T_Cbsi = 0;
for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    dod     =   hmrBandpassFilt(dod,t,0,0.5);
    dc      =   hmrOD2Conc(dod,SD1,[6  6]);
    tic
    [dc_Cbsi] = hmrMotionCorrectCbsi(dc,SD1,0);
    T_Cbsi = T_Cbsi + toc;
    
end
fprintf('Time for Cbsi is %f\n',T_Cbsi)
%% Spline 99
fprintf('Spline 99\n')
T_Spline = 0;
for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod                 =   hmrConc2OD( dc, SD1, ppf );
    tic
    [~,tIncChAuto_before_Spline]    =   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,200);
    [dod_Spline]                    =   hmrMotionCorrectSpline(dod,t,SD1,tIncChAuto_before_Spline,0.99);
    T_Spline                        =   T_Spline + toc;
end
fprintf('Time for Spline is %f\n',T_Spline)
%% PCA99
fprintf('PCA99\n')
T_PCA99 = 0;
sigma = 0.99;

for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    tic
    [dod_PCA,~,~,~,~]   =   hmrMotionCorrectPCArecurse(dod,t,SD1,tIncMan,0.5,1,STD,200,sigma,5);
    T_PCA99 = T_PCA99 + toc;
    
end
fprintf('Time for PCA99 is %f\n',T_PCA99)

%% PCA50
fprintf('PCA50\n')
T_PCA50 = 0;
sigma = 0.50;

for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    tic
    [dod_PCA,~,~,~,~]   =   hmrMotionCorrectPCArecurse(dod,t,SD1,tIncMan,0.5,1,STD,200,sigma,5);
    T_PCA50 = T_PCA50 + toc;
end
fprintf('Time for PCA50 is %f\n',T_PCA50)
%% Kalman
fprintf('Kalman\n')
T_Kalman = 0;
for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    y               =   dod;
    oscFreq         =   [0,0.01,0.001,0.0001];
    xo              =   ones(1,length(oscFreq)+1)*y(1,1);
    Po              =   ones(1,length(oscFreq)+1)*(y(1,1)^2);
    Qo              =   zeros(1,length(oscFreq)+1);
    hrfParam        =   [2 2];
    s = [1;zeros(size(y,1)-1,1)];
    tic
    [x, yStim,dod_Kalman,y,C,Q] = hmrKalman2( y, s, t, xo, Po, Qo, 'box', hrfParam, oscFreq );
    T_Kalman = T_Kalman + toc;
    
end
fprintf('Time for Kalman is %f\n',T_Kalman)
%% Wavelet35
fprintf('Wavelet35\n')
T_Wavelet35 = 0;
for i = 1:N
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    tic
    [dod_Wavelet35]       =   hmrMotionCorrectWavelet(dod,SD1,0.35);
    T_Wavelet35 = T_Wavelet35 + toc;
    
end
fprintf('Time for Wavelet35 is %f\n',T_Wavelet35)
%% Wavelet05
fprintf('Wavelet05\n')
T_Wavelet05 = 0;
for i = 1:100
    dc_HbO  =   HbO_noised(i,:);
    dc_HbR  =   HbR_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dc_real = [HbO(i,:),HbR(i,:)];

    dod     =   hmrConc2OD( dc, SD1, ppf );
    tic
    [dod_Wavelet05]       =   hmrMotionCorrectWavelet(dod,SD1,0.05);
    T_Wavelet05 = T_Wavelet05 + toc;
    
end
fprintf('Time for Wavelet05 is %f\n',T_Wavelet05)
%% output time






% T_Spline = 8.637879;
% T_Wavelet05 = 1111.117784;
% T_Wavelet35 = 1102.784217;
% T_Kalman = 55.251534;
% T_PCA99 = 6.714928;
% T_PCA50 = 7.102599;
% T_Cbsi = 0.461004;
T_DAE = 0.3288;
% T_DAE = 0.3555;for 4 layers
fprintf('Time for DAE is %f\n',T_DAE)

T_list = [T_Spline,T_Wavelet05,T_Wavelet35,T_Kalman,T_PCA99,T_PCA50,T_Cbsi,T_DAE];
labels = {'Spline','Wavelet05','Wavelet35','Kalman','PCA99','PCA50','Cbsi','DAE'};
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
box off