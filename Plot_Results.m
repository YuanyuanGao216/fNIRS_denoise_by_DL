function Plot_Results()
%% load data
load('Processed_data/Real_HbO.mat');
load('Processed_data/HbO_Spline.mat','Spline')
load('Processed_data/HbO_Wavelet.mat','Wavelet')
load('Processed_data/HbO_Kalman.mat','Kalman')
load('Processed_data/HbO_PCA97.mat','HbO_PCA97')
load('Processed_data/HbO_PCA80.mat','HbO_PCA80')
if exist('Processed_data/HbO_NN.mat', 'file') == 2
    load('Processed_data/HbO_NN.mat','HbO_NN')% File exists.
else
    
    % File does not exist.
end

load('Processed_data/Real_HbR.mat');
load('Processed_data/HbR_Spline.mat','Spline')
load('Processed_data/HbR_Wavelet.mat','Wavelet')
load('Processed_data/HbR_Kalman.mat','Kalman')
load('Processed_data/HbR_PCA97.mat','HbO_PCA97')
load('Processed_data/HbR_PCA80.mat','HbO_PCA80')
if exist('Processed_data/HbR_NN.mat', 'file') == 2
    load('Processed_data/HbR_NN.mat','HbO_NN')% File exists.
else
    HbR_NN = zeros(size(Real_HbR));
    % File does not exist.
end

load('Processed_data/MA_list.mat','MA_list')
%% figure 1: Example plot
piece = input('Which piece you want to show? ');

figure('Renderer', 'painters','Position',[10 10 600 250]);
subplot(1,2,1)
plot(Real_HbO(piece,:),'k-','DisplayName','No correction');hold on;
plot(HbO_Spline(piece,:),'b-','DisplayName','Spline');hold on;
plot(HbO_Wavelet(piece,:),'g-','DisplayName','Wavelet');hold on;
plot(HbO_Kalman(piece,:),'y-','DisplayName','Kalman');hold on;
plot(HbO_PCA97(piece,:),'m-','DisplayName','PCA97');hold on;
plot(HbO_PCA80(piece,:),'c-','DisplayName','PCA80');hold on;
plot(HbO_NN(piece,:),'r-','DisplayName','DAE');hold on;
ylabel('/delta HbO (/muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs)
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
subplot(1,2,2)
plot(Real_HbR(piece,:),'k-','DisplayName','No correction');hold on;
plot(HbR_Spline(piece,:),'b-','DisplayName','Spline');hold on;
plot(HbR_Wavelet(piece,:),'g-','DisplayName','Wavelet');hold on;
plot(HbR_Kalman(piece,:),'y-','DisplayName','Kalman');hold on;
plot(HbR_PCA97(piece,:),'m-','DisplayName','PCA97');hold on;
plot(HbR_PCA80(piece,:),'c-','DisplayName','PCA80');hold on;
plot(HbR_NN(piece,:),'r-','DisplayName','DAE');hold on;
ylabel('/delta HbR (/muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs)
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
%% figure 2: Bar plots with the number of trials with MA
n_NN_HbO = 0;
n_NN_HbR = 0;
for i = 1:size(HbO_NN,2)
    dc_HbO = HbO_NN(:,i);
    dc_HbR = HbR_NN(:,i);
    dc = [dc_HbO,dc_HbR];
    SD1.MeasList = [1,1,1,1;1,1,1,2];
    SD1.MeasListAct = [1 1];
    SD1.Lambda = [760;850];
    ppf = [6,6];
    dod = hmrConc2OD( dc, SD, ppf );
    fs = 7.8125;
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod1, t, SD1, ones(size(d_1)), 0.5, 1, 30, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_NN_HbO = n_NN_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR+n_MA;
end
MA_list = [MA_list,[n_NN_HbO;n_NN_HbR]];
bar(MA_list)
%% boxplot for AUC0-2, AUCratio X (No_correction,Rejection, PCA97, PCA80, Spline, wavelet, kalman, NN)

figure

subplot(2,2,1)

