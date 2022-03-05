close all

%% add homer path

pathHomer   =   '../../Tools/homer2_src_v2_3_10202017';
oldpath     =   cd(pathHomer);
setpaths;
cd(oldpath);
%% load data
labels = {'No correction','Spline','Wavelet05','Wavelet35','Kalman','PCA99','PCA50','Cbsi','DAE'};

load('Processed_data/Real_HbO.mat');
load('Processed_data/HbO_Spline.mat',       'HbO_Spline')
load('Processed_data/HbO_Wavelet05.mat',    'HbO_Wavelet05')
load('Processed_data/HbO_Wavelet35.mat',    'HbO_Wavelet35')
load('Processed_data/HbO_Kalman.mat',       'HbO_Kalman')
load('Processed_data/HbO_PCA99.mat',        'HbO_PCA99')
load('Processed_data/HbO_PCA50.mat',        'HbO_PCA50')
load('Processed_data/HbO_Cbsi.mat',         'HbO_Cbsi')

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
load('Processed_data/HbR_Wavelet35.mat',    'HbR_Wavelet35')
load('Processed_data/HbR_Kalman.mat',       'HbR_Kalman')
load('Processed_data/HbR_PCA99.mat',        'HbR_PCA99')
load('Processed_data/HbR_PCA50.mat',        'HbR_PCA50')
load('Processed_data/HbR_Cbsi.mat',         'HbR_Cbsi')

load('Processed_data/MA_list.mat','MA_list')

SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];
fs = 7.8125;
STD = 10;

%% figure 2: Bar plots with the number of trials with MA, no backforth

n_NN_HbO = 0;
n_NN_HbR = 0;
for i = 3:size(HbO_NN,1)
    dc_HbO = HbO_NN(i,:);
    dc_HbR = HbR_NN(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, 20, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    if n_MA ~= 0
        fprintf('%d\n',i)
    end
    n_NN_HbO = n_NN_HbO + n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR + n_MA;
end
MA_list_new = [MA_list(1,:),n_NN_HbO];
MA_list_percetage = MA_list_new./MA_list_new(1);

figure
% b = bar(MA_list_new(1,:),'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);
b = bar(MA_list_percetage,'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);

ylabel({'Residual motion artifacts'})
set(gca, 'XTick', 1:size(MA_list_new,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
yticks([0 1])
yticklabels({'0','100%'})
ylim([0 1.5])
xlim([0 10])

xtips1  =   b(1).XEndPoints;
ytips1  =   b(1).YEndPoints;
per_label = b(1).YData;

for i = 1:length(per_label)
    x = per_label(i)*100;
    label = sprintf('%.0f%%',x);
    text(xtips1(i),ytips1(i),label,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end
% labels1 =   [string(b(1).YData*100),'%'];

set(gcf,'Position',[3   490   330   215]);
box off
fprintf('%d\n',MA_list(1,end))
MA_list(1,:)
return
%% figure 1: Example plot
piece = input('Which piece you want to show? ');
% 130 and 1

figure('Renderer', 'painters','Position',[72        1229         989         196]);
subplot(1,2,1)
plot(Real_HbO(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbO_Spline(piece,:)*1e6,'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
plot(HbO_Wavelet05(piece,:)*1e6,'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
plot(HbO_Wavelet35(piece,:)*1e6,'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1.5);hold on;
plot(HbO_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
plot(HbO_PCA99(piece,:)*1e6,'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
plot(HbO_PCA50(piece,:)*1e6,'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
plot(HbO_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot(HbO_NN(piece,:)*1e6,'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
ylabel('\Delta HbO (\muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs)
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)

subplot(1,2,2)
plot(Real_HbR(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbR_Spline(piece,:)*1e6,'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
plot(HbR_Wavelet05(piece,:)*1e6,'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
plot(HbR_Wavelet35(piece,:)*1e6,'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1.5);hold on;
plot(HbR_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
plot(HbR_PCA99(piece,:)*1e6,'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
plot(HbR_PCA50(piece,:)*1e6,'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
plot(HbR_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot(HbR_NN(piece,:)*1e6,'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
ylabel('\Delta HbR (\muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs) 
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)


%% boxplot for AUC0-18, AUCratio = AUC0-18/AUC18-26 
% AUC0-2
t1 = 2;
AUC0_2_HbO      = zeros(9,size(Real_HbO,1));
AUC0_2_HbO(1,:) = AUC(Real_HbO,0,t1);
AUC0_2_HbO(2,:) = AUC(HbO_Spline,0,t1);
AUC0_2_HbO(3,:) = AUC(HbO_Wavelet05,0,t1);
AUC0_2_HbO(4,:) = AUC(HbO_Wavelet35,0,t1);
AUC0_2_HbO(5,:) = AUC(HbO_Kalman,0,t1);
AUC0_2_HbO(6,:) = AUC(HbO_PCA99,0,t1);
AUC0_2_HbO(7,:) = AUC(HbO_PCA50,0,t1);
AUC0_2_HbO(8,:) = AUC(HbO_Cbsi,0,t1);
AUC0_2_HbO(9,:) = AUC(HbO_NN,0,t1);

AUC0_2_HbR      = zeros(9,size(Real_HbO,1));
AUC0_2_HbR(1,:) = AUC(Real_HbR,0,t1);
AUC0_2_HbR(2,:) = AUC(HbR_Spline,0,t1);
AUC0_2_HbR(3,:) = AUC(HbR_Wavelet05,0,t1);
AUC0_2_HbR(4,:) = AUC(HbR_Wavelet35,0,t1);
AUC0_2_HbR(5,:) = AUC(HbR_Kalman,0,t1);
AUC0_2_HbR(6,:) = AUC(HbR_PCA99,0,t1);
AUC0_2_HbR(7,:) = AUC(HbR_PCA50,0,t1);
AUC0_2_HbR(8,:) = AUC(HbR_Cbsi,0,t1);
AUC0_2_HbR(9,:) = AUC(HbR_NN,0,t1);

% AUC 2-4
t2 = 7;
AUC2_4_HbO      = zeros(9,size(Real_HbO,1));
AUC2_4_HbO(1,:) = AUC(Real_HbR,t1,t2);
AUC2_4_HbO(2,:) = AUC(HbO_Spline,t1,t2);
AUC2_4_HbO(3,:) = AUC(HbO_Wavelet05,t1,t2);
AUC2_4_HbO(4,:) = AUC(HbO_Wavelet35,t1,t2);
AUC2_4_HbO(5,:) = AUC(HbO_Kalman,t1,t2);
AUC2_4_HbO(6,:) = AUC(HbO_PCA99,t1,t2);
AUC2_4_HbO(7,:) = AUC(HbO_PCA50,t1,t2);
AUC2_4_HbO(8,:) = AUC(HbO_Cbsi,t1,t2);
AUC2_4_HbO(9,:) = AUC(HbO_NN,t1,t2);

AUC2_4_HbR      = zeros(9,size(Real_HbO,1));
AUC2_4_HbR(1,:) = AUC(Real_HbR,t1,t2);
AUC2_4_HbR(2,:) = AUC(HbR_Spline,t1,t2);
AUC2_4_HbR(3,:) = AUC(HbR_Wavelet05,t1,t2);
AUC2_4_HbR(4,:) = AUC(HbR_Wavelet35,t1,t2);
AUC2_4_HbR(5,:) = AUC(HbR_Kalman,t1,t2);
AUC2_4_HbR(6,:) = AUC(HbR_PCA99,t1,t2);
AUC2_4_HbR(7,:) = AUC(HbR_PCA50,t1,t2);
AUC2_4_HbR(8,:) = AUC(HbR_Cbsi,t1,t2);
AUC2_4_HbR(9,:) = AUC(HbR_NN,t1,t2);

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
%% table
mean_AUC0_2_HbO = nanmean(AUC0_2_HbO * 1e6,2);
std_AUC0_2_HbO = nanstd(AUC0_2_HbO * 1e6,0,2);
fprintf('HbO AUC0_2:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_AUC0_2_HbO(i),std_AUC0_2_HbO(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(AUC0_2_HbO(1,:),AUC0_2_HbO(i,:));
    fprintf('p = %.5f\n',p)
end
%% 
mean_AUCratio_HbO = nanmean(AUCratio_HbO,2);
std_AUCratio_HbO = nanstd(AUCratio_HbO,0,2);
fprintf('HbO AUCratio_HbO:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_AUCratio_HbO(i),std_AUCratio_HbO(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(AUCratio_HbO(1,:),AUCratio_HbO(i,:));
    fprintf('p = %.5f\n',p)
end
%%
mean_AUC0_2_HbR = nanmean(AUC0_2_HbR * 1e6,2);
std_AUC0_2_HbR = nanstd(AUC0_2_HbR * 1e6,0,2);
fprintf('HbR AUC0_2:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_AUC0_2_HbR(i),std_AUC0_2_HbR(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(AUC0_2_HbR(1,:),AUC0_2_HbR(i,:));
    fprintf('p = %.5f\n',p)
end
%% 
mean_AUCratio_HbR = nanmean(AUCratio_HbR,2);
std_AUCratio_HbR = nanstd(AUCratio_HbR,0,2);
fprintf('HbO AUCratio_HbO:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_AUCratio_HbR(i),std_AUCratio_HbR(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(AUCratio_HbR(1,:),AUCratio_HbR(i,:));
    fprintf('p = %.5f\n',p)
end
return
%% figure 4 dot plot contrast
fontsize = 12;
figure
for i = 1:8
    subplot(4,2,i)
    x_data  =   AUC0_2_HbO(1,:) * 1e6;
    y_data  =   AUC0_2_HbO(i+1,:) * 1e6;
    plot(x_data,y_data,'bx');hold on;
    axis square
    n1      =   sum(y_data>x_data);
    n2      =   sum(y_data<x_data);
    n       =   size(x_data,2);
    n1      =   n1/n;
    n2      =   n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl      =   xlim;
    yl      =   ylim;
    n1_x    =   xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
    n1_y    =   (yl(1)+(yl(2)-yl(1))*3/4);
    n1_str  =   sprintf('%.2f%%',n1*100);
    text(n1_x,n1_y,n1_str);
    n2_x    =   (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y    =   (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str  =   sprintf('%.2f%%',n2*100);
    text(n2_x,n2_y,n2_str)
    xlabel('No correction','FontSize', fontsize)
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[30 10 300 650])

figure
for i = 1:8
    subplot(4,2,i)
    x_data = AUC0_2_HbR(1,:) * 1e6;
    y_data = AUC0_2_HbR(i+1,:) * 1e6;
    plot(x_data,y_data,'bx');hold on;
    axis square
    n1  =   sum(y_data>x_data);
    n2  =   sum(y_data<x_data);
    n   =   size(x_data,2);
    n1  =   n1/n;
    n2  =   n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl = xlim;
    yl = ylim;
    n1_x    =   xl(1) + (xl(2) - xl(1)) * 1/4 - (xl(2) - xl(1)) * 1/8;
    n1_y    =   (yl(1) + (yl(2) - yl(1)) * 3/4);
    n1_str  =   sprintf('%.2f%%',n1 * 100);
    text(n1_x,n1_y,n1_str);
    n2_x    =   (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y    =   (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str  =   sprintf('%.2f%%',n2*100);
    text(n2_x,n2_y,n2_str)
    xlabel('No correction','FontSize', fontsize)
    ylabel(labels{i + 1},'FontSize', fontsize)
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
    n1  =   sum(y_data>x_data);
    n2  =   sum(y_data<x_data);
    n   =   size(x_data,2);
    n1  =   n1/n;
    n2  =   n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl      =   xlim;
    yl      =   ylim;
    n1_x    =   xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
    n1_y    =   (yl(1)+(yl(2)-yl(1))*3/4);
    n1_str  = sprintf('%.2f%%',n1*100);
    text(n1_x,n1_y,n1_str);
    n2_x    = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y    = (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str  = sprintf('%.2f%%',n2*100);
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
    n1  =   sum(y_data>x_data);
    n2  =   sum(y_data<x_data);
    n   =   size(x_data,2);
    n1  =   n1/n;
    n2  =   n2/n;
    min_data = min([min(x_data),min(y_data)]);
    max_data = max([max(x_data),max(y_data)]);
    plot([min_data,max_data],[min_data,max_data],'r-')
    xlim([min_data max_data])
    ylim([min_data max_data])
    xl = xlim;
    yl = ylim;
    n1_x    =   xl(1)+(xl(2)-xl(1))*1/4-(xl(2)-xl(1))*1/8;
    n1_y    =   (yl(1)+(yl(2)-yl(1))*3/4);
    n1_str  =   sprintf('%.2f%%',n1*100);
    text(n1_x,n1_y,n1_str);
    n2_x = (xl(1)+(xl(2)-xl(1))*3/4)-(xl(2)-xl(1))*1/8;
    n2_y = (yl(1)+(yl(2)-yl(1))*1/4);
    n2_str = sprintf('%.2f%%',n2*100);
    text(n2_x,n2_y,n2_str)
    xlabel('No correction','FontSize', fontsize)
    ylabel(labels{i+1},'FontSize', fontsize)
end

set(gcf,'Position',[350 10 300 650])
%% figure 2.1: Bar plots with the number of trials with MA

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
    n_NN_HbO = n_NN_HbO + n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR + n_MA;
end

n_Wavelet05_HbO = 0;
n_Wavelet05_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_Wavelet05(i,:);
    dc_HbR  =   HbR_Wavelet05(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]      =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]          =   CalMotionArtifact(tIncChAuto(:,1));
    n_Wavelet05_HbO     =   n_Wavelet05_HbO+n_MA;
    [n_MA,~,~]          =   CalMotionArtifact(tIncChAuto(:,2));
    n_Wavelet05_HbR     =   n_Wavelet05_HbR+n_MA;
end

n_Wavelet35_HbO = 0;
n_Wavelet35_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_Wavelet35(i,:);
    dc_HbR  =   HbR_Wavelet35(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_Wavelet35_HbO =   n_Wavelet35_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_Wavelet35_HbR =   n_Wavelet35_HbR+n_MA;
end

n_Spline_HbO = 0;
n_Spline_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_Spline(i,:);
    dc_HbR  =   HbR_Spline(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_Spline_HbO    =   n_Spline_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_Spline_HbR    =   n_Spline_HbR+n_MA;
end

n_Kalman_HbO = 0;
n_Kalman_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_Kalman(i,:);
    dc_HbR  =   HbR_Kalman(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_Kalman_HbO    =   n_Kalman_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_Kalman_HbR    =   n_Kalman_HbR+n_MA;
end

n_PCA99_HbO = 0;
n_PCA99_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_PCA99(i,:);
    dc_HbR  =   HbR_PCA99(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_PCA99_HbO     =   n_PCA99_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_PCA99_HbR     =   n_PCA99_HbR+n_MA;
end

n_PCA50_HbO = 0;
n_PCA50_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_PCA50(i,:);
    dc_HbR  =   HbR_PCA50(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_PCA50_HbO     =   n_PCA50_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_PCA50_HbR     =   n_PCA50_HbR+n_MA;
end

n_Cbsi_HbO = 0;
n_Cbsi_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO  =   HbO_Cbsi(i,:);
    dc_HbR  =   HbR_Cbsi(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    dod     =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_Cbsi_HbO      =   n_Cbsi_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_Cbsi_HbR      =   n_Cbsi_HbR+n_MA;
end

n_MA_total_HbO = 0;
n_MA_total_HbR = 0;

for i = 1:size(HbO_NN,1)
    dc_HbO          =   Real_HbO(i,:);
    dc_HbR          =   Real_HbR(i,:);
    dc              =   [dc_HbO;dc_HbR]';
    dod             =   hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,1));
    n_MA_total_HbO  =   n_MA_total_HbO+n_MA;
    [n_MA,~,~]      =   CalMotionArtifact(tIncChAuto(:,2));
    n_MA_total_HbR  =   n_MA_total_HbR+n_MA;
end
MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet35_HbO,...
    n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet35_HbR,...
    n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];


figure
b = bar(MA_list(1,:),'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);

ylabel('No. of Motion Artifacts')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylim([0 600])
xlim([0 10])

xtips1  =   b(1).XEndPoints;
ytips1  =   b(1).YEndPoints;
labels1 =   string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[3   490   330   215]);
box off
fprintf('%d\n',MA_list(1,end))
MA_list(1,:)
%%
function AUCvalue = AUC(Hb,t1,t2)
if t2 <= t1
    fprintf('t2 is lower than t1\n')
    return
end
fs = 7.8125;
Hb = Hb - repmat(mean(Hb(:,1:round(fs*1)),2),1,512);
AUCvalue= abs(trapz(Hb(:,round(fs*t1)+1:round(fs*(t2)))./fs,2));
end