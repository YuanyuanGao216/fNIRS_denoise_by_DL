% function Plot_Results()
%% load data
load('Processed_data/Real_HbO.mat');
load('Processed_data/HbO_Spline.mat',   'HbO_Spline')
load('Processed_data/HbO_Wavelet.mat',  'HbO_Wavelet')
load('Processed_data/HbO_Kalman.mat',   'HbO_Kalman')
load('Processed_data/HbO_PCA97.mat',    'HbO_PCA97')
load('Processed_data/HbO_PCA80.mat',    'HbO_PCA80')
load('Processed_data/HbO_Cbsi.mat',     'HbO_Cbsi')

if exist('Processed_data/HbO_NN.mat', 'file') == 2
    load('Processed_data/HbO_NN.mat','HbO_NN')% File exists.
else
    HbO_NN = zeros(size(Real_HbO));
    % File does not exist.
end

load('Processed_data/Real_HbR.mat');
load('Processed_data/HbR_Spline.mat',   'HbR_Spline')
load('Processed_data/HbR_Wavelet.mat',  'HbR_Wavelet')
load('Processed_data/HbR_Kalman.mat',   'HbR_Kalman')
load('Processed_data/HbR_PCA97.mat',    'HbR_PCA97')
load('Processed_data/HbR_PCA80.mat',    'HbR_PCA80')
load('Processed_data/HbR_Cbsi.mat',     'HbR_Cbsi')
if exist('Processed_data/HbR_NN.mat', 'file') == 2
    load('Processed_data/HbR_NN.mat','HbR_NN')% File exists.
else
    HbR_NN = zeros(size(Real_HbR));
    % File does not exist.
end

load('Processed_data/MA_list.mat','MA_list')
%% figure 1: Example plot
piece = input('Which piece you want to show? ');

figure('Renderer', 'painters','Position',[10 10 1200 250]);
subplot(1,2,1)
plot(Real_HbO(piece,:),'k-','DisplayName','No correction');hold on;
plot(HbO_Spline(piece,:),'b-','DisplayName','Spline');hold on;
plot(HbO_Wavelet(piece,:),'g-','DisplayName','Wavelet');hold on;
plot(HbO_Kalman(piece,:),'y-','DisplayName','Kalman');hold on;
plot(HbO_PCA97(piece,:),'m-','DisplayName','PCA97');hold on;
plot(HbO_PCA80(piece,:),'c-','DisplayName','PCA80');hold on;
plot(HbO_Cbsi(piece,:),'Color',[153, 102, 51]./255,'DisplayName','Cbsi');hold on;
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
plot(HbR_Cbsi(piece,:),'Color',[153, 102, 51]./255,'DisplayName','Cbsi');hold on;
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
SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];
fs = 7.8125;
STD = 10;
for i = 1:size(HbO_NN,2)
    dc_HbO = HbO_NN(:,i);
    dc_HbR = HbR_NN(:,i);
    dc = [dc_HbO,dc_HbR];
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_NN_HbO = n_NN_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR+n_MA;
end
MA_list = [MA_list,[n_NN_HbO;n_NN_HbR]];
figure
bar(MA_list(1,:))
%% boxplot for AUC0-2, AUCratio 
% AUC0-2
AUC0_2_HbO = zeros(8,size(Real_HbO,1));
AUC0_2_HbO(1,:) = abs(trapz(Real_HbO(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(2,:) = abs(trapz(HbO_Spline(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(3,:) = abs(trapz(HbO_Wavelet(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(4,:) = abs(trapz(HbO_Kalman(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(5,:) = abs(trapz(HbO_PCA97(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(6,:) = abs(trapz(HbO_PCA80(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(7,:) = abs(trapz(HbO_Cbsi(:,1:round(fs*2))./fs,2));
AUC0_2_HbO(8,:) = abs(trapz(HbO_NN(:,1:round(fs*2))./fs,2));

AUC0_2_HbR = zeros(7,size(Real_HbO,1));
AUC0_2_HbR(1,:) = abs(trapz(Real_HbR(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(2,:) = abs(trapz(HbR_Spline(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(3,:) = abs(trapz(HbR_Wavelet(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(4,:) = abs(trapz(HbR_Kalman(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(5,:) = abs(trapz(HbR_PCA97(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(6,:) = abs(trapz(HbR_PCA80(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(7,:) = abs(trapz(HbR_Cbsi(:,1:round(fs*2))./fs,2));
AUC0_2_HbR(8,:) = abs(trapz(HbR_NN(:,1:round(fs*2))./fs,2));
% AUC 2-4
AUC2_4_HbO = zeros(8,size(Real_HbO,1));
AUC2_4_HbO(1,:) = abs(trapz(Real_HbO(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(2,:) = abs(trapz(HbO_Spline(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(3,:) = abs(trapz(HbO_Wavelet(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(4,:) = abs(trapz(HbO_Kalman(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(5,:) = abs(trapz(HbO_PCA97(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(6,:) = abs(trapz(HbO_PCA80(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(7,:) = abs(trapz(HbO_Cbsi(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbO(8,:) = abs(trapz(HbO_NN(:,round(fs*2):round(fs*4))./fs,2));

AUC2_4_HbR = zeros(7,size(Real_HbO,1));
AUC2_4_HbR(1,:) = abs(trapz(Real_HbR(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(2,:) = abs(trapz(HbR_Spline(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(3,:) = abs(trapz(HbR_Wavelet(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(4,:) = abs(trapz(HbR_Kalman(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(5,:) = abs(trapz(HbR_PCA97(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(6,:) = abs(trapz(HbR_PCA80(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(7,:) = abs(trapz(HbR_Cbsi(:,round(fs*2):round(fs*4))./fs,2));
AUC2_4_HbR(8,:) = abs(trapz(HbR_NN(:,round(fs*2):round(fs*4))./fs,2));

% AUC ratio
AUCratio_HbO = AUC2_4_HbO./AUC0_2_HbO;
AUCratio_HbR = AUC2_4_HbR./AUC0_2_HbR;

figure
subplot(2,2,1)
boxplot(AUC0_2_HbO')
title('AUC_(0-2) HbO')
subplot(2,2,2)
boxplot(AUCratio_HbO')
title('AUC_(ratio) HbO')
subplot(2,2,3)
boxplot(AUC0_2_HbR')
title('AUC_(0-2) HbR')
subplot(2,2,4)
title('AUC_(ratio) HbR')
boxplot(AUCratio_HbR')
%% figure 4 dot plot contrast
fontsize = 8;
label = {'Spline','Wavelet','Kalman','PCA97','PCA80','Cbsi','DAE'};
figure
for i = 1:7
    subplot(4,2,i)
    x_data = AUC0_2_HbO(1,:);
    y_data = AUC0_2_HbO(i+1,:);
    plot(x_data,y_data,'bx');hold on;
    axis square
    n1 = sum(y_data>x_data);
    n2 = sum(y_data<x_data);
    n = size(x_data,2);
    n1 = n1/n;
    n2 = n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl = xlim;
    yl = ylim;
    n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
    n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
    n1_str = sprintf('%.2f%%',n1*100);
    text(n1_x,n1_y,n1_str);
    n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str = sprintf('%.2f%%',n2*100);
    text(n2_x,n2_y,n2_str)
    xlabel('No correction','FontSize', fontsize)
    ylabel(label{i},'FontSize', fontsize)
end

set(gcf,'Position',[30 10 300 650])

figure
for i = 1:7
    subplot(4,2,i)
    x_data = AUC0_2_HbR(1,:);
    y_data = AUC0_2_HbR(i+1,:);
    plot(x_data,y_data,'bx');hold on;
    axis square
    n1 = sum(y_data>x_data);
    n2 = sum(y_data<x_data);
    n = size(x_data,2);
    n1 = n1/n;
    n2 = n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl = xlim;
    yl = ylim;
    n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
    n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
    n1_str = sprintf('%.2f%%',n1*100);
    text(n1_x,n1_y,n1_str);
    n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str = sprintf('%.2f%%',n2*100);
    text(n2_x,n2_y,n2_str)
    xlabel('No correction','FontSize', fontsize)
    ylabel(label{i},'FontSize', fontsize)
end

set(gcf,'Position',[350 10 300 650])