% function Plot_Results()
clear all
close all
%% add homer path

pathHomer = '../../Tools/homer2_src_v2_3_10202017';

oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% load data
labels = {'No correction','Spline','Wavelet05','Wavelet40','Kalman','PCA99','PCA50','Cbsi','DAE'};

load('Processed_data/Real_HbO.mat');
load('Processed_data/HbO_Spline.mat',   'HbO_Spline')
load('Processed_data/HbO_Wavelet05.mat',  'HbO_Wavelet05')
load('Processed_data/HbO_Wavelet40.mat',  'HbO_Wavelet40')
load('Processed_data/HbO_Kalman.mat',   'HbO_Kalman')
load('Processed_data/HbO_PCA99.mat',    'HbO_PCA99')
load('Processed_data/HbO_PCA50.mat',    'HbO_PCA50')
load('Processed_data/HbO_Cbsi.mat',     'HbO_Cbsi')

filepath = 'Processed_data/Real_NN_8layers_save.mat';

if exist(filepath, 'file') == 2
    load(filepath)% File exists.
    Hb_NN = Y_real;
else
    Hb_NN = zeros(size(Real_HbO));
    % File does not exist.
end
HbO_NN = Hb_NN(:,1:size(Real_HbO,2));
HbR_NN = Hb_NN(:,size(Real_HbO,2)+1:end);

load('Processed_data/Real_HbR.mat');
load('Processed_data/HbR_Spline.mat',       'HbR_Spline')
load('Processed_data/HbR_Wavelet05.mat',    'HbR_Wavelet05')
load('Processed_data/HbR_Wavelet40.mat',    'HbR_Wavelet40')
load('Processed_data/HbR_Kalman.mat',       'HbR_Kalman')
load('Processed_data/HbR_PCA99.mat',        'HbR_PCA99')
load('Processed_data/HbR_PCA50.mat',        'HbR_PCA50')
load('Processed_data/HbR_Cbsi.mat',         'HbR_Cbsi')

load('Processed_data/MA_list.mat','MA_list')
%% figure 1: Example plot
piece = input('Which piece you want to show? ');
% 130

figure('Renderer', 'painters','Position',[10 10 1200 250]);
subplot(1,2,1)
plot(Real_HbO(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbO_Spline(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
plot(HbO_Wavelet05(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
plot(HbO_Wavelet40(piece,:),'color','g','linestyle',':','DisplayName','Wavelet40','LineWidth',1.5);hold on;
plot(HbO_Kalman(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
plot(HbO_PCA99(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
plot(HbO_PCA50(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
plot(HbO_Cbsi(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot(HbO_NN(piece,:),'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
ylabel('\Delta HbO (Mol)')
fs = 7.8125;
xticks([0 20 40 60]*fs)
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)

subplot(1,2,2)
plot(Real_HbR(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbR_Spline(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
plot(HbR_Wavelet05(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
plot(HbR_Wavelet40(piece,:),'color','g','linestyle',':','DisplayName','Wavelet40','LineWidth',1.5);hold on;
plot(HbR_Kalman(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
plot(HbR_PCA99(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
plot(HbR_PCA50(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
plot(HbR_Cbsi(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot(HbR_NN(piece,:),'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
ylabel('\Delta HbR (Mol)')
fs = 7.8125;
xticks([0 20 40 60]*fs) 
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)
%% figure 2: Bar plots with the number of trials with MA

SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];
fs = 7.8125;
STD = 10;

n_NN_HbO = 0;
n_NN_HbR = 0;
for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_NN(i,:);
    dc_HbR = HbR_NN(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    if n_MA ~= 0
        fprintf('%d\n',i)
    end
    n_NN_HbO = n_NN_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR+n_MA;
end

n_Wavelet05_HbO = 0;
n_Wavelet05_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_Wavelet05(i,:);
    dc_HbR = HbR_Wavelet05(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_Wavelet05_HbO = n_Wavelet05_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_Wavelet05_HbR = n_Wavelet05_HbR+n_MA;
end

n_Wavelet40_HbO = 0;
n_Wavelet40_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_Wavelet40(i,:);
    dc_HbR = HbR_Wavelet40(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_Wavelet40_HbO = n_Wavelet40_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_Wavelet40_HbR = n_Wavelet40_HbR+n_MA;
end

n_Spline_HbO = 0;
n_Spline_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_Spline(i,:);
    dc_HbR = HbR_Spline(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_Spline_HbO = n_Spline_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_Spline_HbR = n_Spline_HbR+n_MA;
end

n_Kalman_HbO = 0;
n_Kalman_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_Kalman(i,:);
    dc_HbR = HbR_Kalman(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_Kalman_HbO = n_Kalman_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_Kalman_HbR = n_Kalman_HbR+n_MA;
end

n_PCA99_HbO = 0;
n_PCA99_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_PCA99(i,:);
    dc_HbR = HbR_PCA99(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_PCA99_HbO = n_PCA99_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_PCA99_HbR = n_PCA99_HbR+n_MA;
end

n_PCA50_HbO = 0;
n_PCA50_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_PCA50(i,:);
    dc_HbR = HbR_PCA50(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_PCA50_HbO = n_PCA50_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_PCA50_HbR = n_PCA50_HbR+n_MA;
end

n_Cbsi_HbO = 0;
n_Cbsi_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_Cbsi(i,:);
    dc_HbR = HbR_Cbsi(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_Cbsi_HbO = n_Cbsi_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_Cbsi_HbR = n_Cbsi_HbR+n_MA;
end

n_MA_total_HbO = 0;
n_MA_total_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO = Real_HbO(i,:);
    dc_HbR = Real_HbR(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    n_MA_total_HbO = n_MA_total_HbO+n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_MA_total_HbR = n_MA_total_HbR+n_MA;
end
MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet40_HbO,n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet40_HbR,n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];


figure
b = bar(MA_list(1,:),'facecolor',[108, 171, 215]./256);

ylabel('No. of Motion Artifacts')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylim([0 600])
xlim([0 10])

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[3   490   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
b.FaceColor = 'flat';
b.CData(9,:) = [200, 14, 80]./255;
box off
fprintf('%d\n',MA_list(1,end))
MA_list(1,:)
%% boxplot for AUC0-18, AUCratio = AUC0-18/AUC18-26 
% AUC0-2
AUC0_2_HbO = zeros(9,size(Real_HbO,1));
AUC0_2_HbO(1,:) = abs(trapz(Real_HbO(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(2,:) = abs(trapz(HbO_Spline(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(3,:) = abs(trapz(HbO_Wavelet05(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(4,:) = abs(trapz(HbO_Wavelet40(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(5,:) = abs(trapz(HbO_Kalman(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(6,:) = abs(trapz(HbO_PCA99(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(7,:) = abs(trapz(HbO_PCA50(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(8,:) = abs(trapz(HbO_Cbsi(:,1:round(fs*18))./fs,2));
AUC0_2_HbO(9,:) = abs(trapz(HbO_NN(:,1:round(fs*18))./fs,2));

AUC0_2_HbR = zeros(9,size(Real_HbO,1));
AUC0_2_HbR(1,:) = abs(trapz(Real_HbR(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(2,:) = abs(trapz(HbR_Spline(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(3,:) = abs(trapz(HbR_Wavelet05(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(4,:) = abs(trapz(HbR_Wavelet40(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(5,:) = abs(trapz(HbR_Kalman(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(6,:) = abs(trapz(HbR_PCA99(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(7,:) = abs(trapz(HbR_PCA50(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(8,:) = abs(trapz(HbR_Cbsi(:,1:round(fs*18))./fs,2));
AUC0_2_HbR(9,:) = abs(trapz(HbR_NN(:,1:round(fs*18))./fs,2));
% AUC 2-4
AUC2_4_HbO = zeros(9,size(Real_HbO,1));
AUC2_4_HbO(1,:) = abs(trapz(Real_HbO(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(2,:) = abs(trapz(HbO_Spline(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(3,:) = abs(trapz(HbO_Wavelet05(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(4,:) = abs(trapz(HbO_Wavelet40(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(5,:) = abs(trapz(HbO_Kalman(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(6,:) = abs(trapz(HbO_PCA99(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(7,:) = abs(trapz(HbO_PCA50(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(8,:) = abs(trapz(HbO_Cbsi(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbO(9,:) = abs(trapz(HbO_NN(:,round(fs*18):round(fs*26))./fs,2));

AUC2_4_HbR = zeros(9,size(Real_HbO,1));
AUC2_4_HbR(1,:) = abs(trapz(Real_HbR(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(2,:) = abs(trapz(HbR_Spline(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(3,:) = abs(trapz(HbR_Wavelet05(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(4,:) = abs(trapz(HbR_Wavelet40(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(5,:) = abs(trapz(HbR_Kalman(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(6,:) = abs(trapz(HbR_PCA99(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(7,:) = abs(trapz(HbR_PCA50(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(8,:) = abs(trapz(HbR_Cbsi(:,round(fs*18):round(fs*26))./fs,2));
AUC2_4_HbR(9,:) = abs(trapz(HbR_NN(:,round(fs*18):round(fs*26))./fs,2));

% AUC ratio
AUCratio_HbO = AUC2_4_HbO./AUC0_2_HbO;
AUCratio_HbR = AUC2_4_HbR./AUC0_2_HbR;

figure
subplot(2,2,1)
boxplot(AUC0_2_HbO','colors','k','OutlierSize',2,'Symbol','b.')
title('AUC0-18 HbO')
set(gca, 'XTick', 1:size(AUC0_2_HbO,1),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylabel('\muMol*s')

subplot(2,2,2)
boxplot(AUCratio_HbO','colors','k','OutlierSize',2,'Symbol','b.')
title('AUCratio HbO')
set(gca, 'XTick',1:size(AUC0_2_HbO,1),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)

subplot(2,2,3)
boxplot(AUC0_2_HbR','colors','k','OutlierSize',2,'Symbol','b.')
title('AUC0-18 HbR')
set(gca, 'XTick',1:size(AUC0_2_HbO,1),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylabel('\muMol*s')

subplot(2,2,4)
boxplot(AUCratio_HbR','colors','k','OutlierSize',2,'Symbol','b.')
title('AUCratio HbR')
set(gca, 'XTick', 1:size(AUC0_2_HbO,1),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylim([0 15])
%% 
[p,~,stats] = anova1(AUC0_2_HbR');
[c,m,h,nms] = multcompare(stats,'Display','on');


%% figure 4 dot plot contrast
fontsize = 12;
figure
for i = 1:8
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
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[30 10 300 650])

figure
for i = 1:8
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
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[350 10 300 650])
%% figure 4 dot plot contrast
figure
for i = 1:8
    subplot(4,2,i)
    x_data = AUCratio_HbO(1,:);
    y_data = AUCratio_HbO(i+1,:);
    loglog(x_data,y_data,'bx');hold on;
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
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[30 10 300 650])

figure
for i = 1:8
    subplot(4,2,i)
    x_data = AUCratio_HbR(1,:);
    y_data = AUCratio_HbR(i+1,:);
    loglog(x_data,y_data,'bx');hold on;
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
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[350 10 300 650])
% %% boxplot for AUC0-2, AUCratio 
% % AUC0-2
% AUC0_2_HbO = zeros(9,size(Real_HbO,1));
% AUC0_2_HbO(1,:) = abs(trapz(Real_HbO(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(2,:) = abs(trapz(HbO_Spline(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(3,:) = abs(trapz(HbO_Wavelet05(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(4,:) = abs(trapz(HbO_Wavelet40(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(5,:) = abs(trapz(HbO_Kalman(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(6,:) = abs(trapz(HbO_PCA99(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(7,:) = abs(trapz(HbO_PCA50(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(8,:) = abs(trapz(HbO_Cbsi(:,1:round(fs*2))./fs,2));
% AUC0_2_HbO(9,:) = abs(trapz(HbO_NN(:,1:round(fs*2))./fs,2));
% 
% AUC0_2_HbR = zeros(9,size(Real_HbO,1));
% AUC0_2_HbR(1,:) = abs(trapz(Real_HbR(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(2,:) = abs(trapz(HbR_Spline(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(3,:) = abs(trapz(HbR_Wavelet05(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(4,:) = abs(trapz(HbR_Wavelet40(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(5,:) = abs(trapz(HbR_Kalman(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(6,:) = abs(trapz(HbR_PCA99(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(7,:) = abs(trapz(HbR_PCA50(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(8,:) = abs(trapz(HbR_Cbsi(:,1:round(fs*2))./fs,2));
% AUC0_2_HbR(9,:) = abs(trapz(HbR_NN(:,1:round(fs*2))./fs,2));
% % AUC 2-4
% AUC2_4_HbO = zeros(9,size(Real_HbO,1));
% AUC2_4_HbO(1,:) = abs(trapz(Real_HbO(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(2,:) = abs(trapz(HbO_Spline(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(3,:) = abs(trapz(HbO_Wavelet05(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(4,:) = abs(trapz(HbO_Wavelet40(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(5,:) = abs(trapz(HbO_Kalman(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(6,:) = abs(trapz(HbO_PCA99(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(7,:) = abs(trapz(HbO_PCA50(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(8,:) = abs(trapz(HbO_Cbsi(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbO(9,:) = abs(trapz(HbO_NN(:,round(fs*2):round(fs*4))./fs,2));
% 
% AUC2_4_HbR = zeros(9,size(Real_HbO,1));
% AUC2_4_HbR(1,:) = abs(trapz(Real_HbR(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(2,:) = abs(trapz(HbR_Spline(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(3,:) = abs(trapz(HbR_Wavelet05(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(4,:) = abs(trapz(HbR_Wavelet40(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(5,:) = abs(trapz(HbR_Kalman(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(6,:) = abs(trapz(HbR_PCA99(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(7,:) = abs(trapz(HbR_PCA50(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(8,:) = abs(trapz(HbR_Cbsi(:,round(fs*2):round(fs*4))./fs,2));
% AUC2_4_HbR(9,:) = abs(trapz(HbR_NN(:,round(fs*2):round(fs*4))./fs,2));
% 
% % AUC ratio
% AUCratio_HbO = AUC2_4_HbO./AUC0_2_HbO;
% AUCratio_HbR = AUC2_4_HbR./AUC0_2_HbR;
% 
% figure
% subplot(2,2,1)
% boxplot(AUC0_2_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('AUC0-2 HbO')
% set(gca, 'XTick', 1:size(AUC0_2_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylabel('\muMol*s')
% 
% subplot(2,2,2)
% boxplot(AUCratio_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('AUCratio HbO')
% set(gca, 'XTick',1:size(AUC0_2_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% 
% subplot(2,2,3)
% boxplot(AUC0_2_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('AUC0-2 HbR')
% set(gca, 'XTick',1:size(AUC0_2_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylabel('\muMol*s')
% 
% subplot(2,2,4)
% boxplot(AUCratio_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('AUCratio HbR')
% set(gca, 'XTick', 1:size(AUC0_2_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylim([0 15])
% %% 
% [p,~,stats] = anova1(AUC0_2_HbR');
% [c,m,h,nms] = multcompare(stats,'Display','off');
% 
% 
% %% figure 4 dot plot contrast
% fontsize = 12;
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = AUC0_2_HbO(1,:);
%     y_data = AUC0_2_HbO(i+1,:);
%     plot(x_data,y_data,'bx');hold on;
%     axis square
%     n1 = sum(y_data>x_data);
%     n2 = sum(y_data<x_data);
%     n = size(x_data,2);
%     n1 = n1/n;
%     n2 = n2/n;
%     min_data = min([min(x_data),min(y_data)]);
%     max_data = max([max(x_data),max(y_data)]);
%     plot([min_data,max_data],[min_data,max_data],'r-')
%     xlim([min_data max_data])
%     ylim([min_data max_data])
%     xl = xlim;
%     yl = ylim;
%     n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
%     n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
%     n1_str = sprintf('%.2f%%',n1*100);
%     text(n1_x,n1_y,n1_str);
%     n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
%     n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
%     n2_str = sprintf('%.2f%%',n2*100);
%     text(n2_x,n2_y,n2_str)
%     xlabel('No correction','FontSize', fontsize)
%     ylabel(labels{i+1},'FontSize', fontsize)
% end
% 
% set(gcf,'Position',[30 10 300 650])
% 
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = AUC0_2_HbR(1,:);
%     y_data = AUC0_2_HbR(i+1,:);
%     plot(x_data,y_data,'bx');hold on;
%     axis square
%     n1 = sum(y_data>x_data);
%     n2 = sum(y_data<x_data);
%     n = size(x_data,2);
%     n1 = n1/n;
%     n2 = n2/n;
%     min_data = min([min(x_data),min(y_data)]);
%     max_data = max([max(x_data),max(y_data)]);
%     plot([min_data,max_data],[min_data,max_data],'r-')
%     xlim([min_data max_data])
%     ylim([min_data max_data])
%     xl = xlim;
%     yl = ylim;
%     n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
%     n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
%     n1_str = sprintf('%.2f%%',n1*100);
%     text(n1_x,n1_y,n1_str);
%     n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
%     n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
%     n2_str = sprintf('%.2f%%',n2*100);
%     text(n2_x,n2_y,n2_str)
%     xlabel('No correction','FontSize', fontsize)
%     ylabel(labels{i+1},'FontSize', fontsize)
% end
% 
% set(gcf,'Position',[350 10 300 650])
% %% figure 4 dot plot contrast
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = AUCratio_HbO(1,:);
%     y_data = AUCratio_HbO(i+1,:);
%     plot(x_data,y_data,'bx');hold on;
%     axis square
%     n1 = sum(y_data>x_data);
%     n2 = sum(y_data<x_data);
%     n = size(x_data,2);
%     n1 = n1/n;
%     n2 = n2/n;
%     min_data = min([min(x_data),min(y_data)]);
%     max_data = max([max(x_data),max(y_data)]);
%     plot([min_data,max_data],[min_data,max_data],'r-')
%     xlim([min_data max_data])
%     ylim([min_data max_data])
%     xl = xlim;
%     yl = ylim;
%     n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
%     n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
%     n1_str = sprintf('%.2f%%',n1*100);
%     text(n1_x,n1_y,n1_str);
%     n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
%     n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
%     n2_str = sprintf('%.2f%%',n2*100);
%     text(n2_x,n2_y,n2_str)
%     xlabel('No correction','FontSize', fontsize)
%     ylabel(labels{i+1},'FontSize', fontsize)
% end
% 
% set(gcf,'Position',[30 10 300 650])
% 
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = AUCratio_HbR(1,:);
%     y_data = AUCratio_HbR(i+1,:);
%     plot(x_data,y_data,'bx');hold on;
%     axis square
%     n1 = sum(y_data>x_data);
%     n2 = sum(y_data<x_data);
%     n = size(x_data,2);
%     n1 = n1/n;
%     n2 = n2/n;
%     min_data = min([min(x_data),min(y_data)]);
%     max_data = max([max(x_data),max(y_data)]);
%     plot([min_data,max_data],[min_data,max_data],'r-')
%     xlim([min_data max_data])
%     ylim([min_data max_data])
%     xl = xlim;
%     yl = ylim;
%     n1_x = xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
%     n1_y = (yl(1)+(yl(2)-yl(1))*3/4);
%     n1_str = sprintf('%.2f%%',n1*100);
%     text(n1_x,n1_y,n1_str);
%     n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
%     n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
%     n2_str = sprintf('%.2f%%',n2*100);
%     text(n2_x,n2_y,n2_str)
%     xlabel('No correction','FontSize', fontsize)
%     ylabel(labels{i+1},'FontSize', fontsize)
% end
% 
% set(gcf,'Position',[350 10 300 650])