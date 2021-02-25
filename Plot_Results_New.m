%% Figure 1 general idea
clear all
close all
clc
define_constants
%% Figure 2 Example
% plot_real_data
% plot_sim_data

%% Figure 3A: The number of residual motion artifacts for the simulated testing dataset.

load('Processed_data/Testing_Spline.mat')
load('Processed_data/Testing_Wavelet01.mat')
load('Processed_data/Testing_Kalman.mat')
load('Processed_data/Testing_PCA99.mat')
load('Processed_data/Testing_PCA97.mat')
load('Processed_data/Testing_Cbsi.mat')
load('Processed_data/Testing_NN.mat')

load('Processed_data/n_MA_list.mat','peak_n_list','shift_n_list','p')

if 1
    n_sample    =   size(peak_n_list,1);
    n_train     =   round(n_sample*0.8);
    n_val       =   round(n_sample*0.1);
    train_idx   =   p(1:n_train);
    val_idx     =   p(n_train+1:n_train+n_val);
    test_idx    =   p(n_train+n_val+1:end);

    peak_n_list_test    =   peak_n_list(test_idx,:);
    shift_n_list_test   =   shift_n_list(test_idx,:);

    n_MA_no_correction      =   sum(peak_n_list_test)+sum(shift_n_list_test);

    
else
    load('Processed_data/Testing_no_correction.mat','n_no_correction')
    n_MA_no_correction = n_no_correction;
end

MA_list = [n_MA_no_correction,...
        n_Spline,...
        n_Wavelet01,...
        n_Kalman,...
        n_PCA99,...
        n_PCA97,...
        n_Cbsi,...
        n_NN];
MA_list_percetage = MA_list./MA_list(1);

figure
b = bar(MA_list_percetage,'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);
labels = {'No correction','Spline','Wavelet01','Kalman','PCA99','PCA797','Cbsi','DAE'};
ylabel({'Residual motion artifacts'})
set(gca, 'XTick', 1:length(MA_list),'fontsize',12)
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
%% Figure 3B: The number of residual motion artifacts for the real dataset.

% save No. of MAs for each files n_files X [n_MA_no_correction,dc_avg_PCA97,n_MA_Spline99,n_MA_Wavelet01,]
load('Processed_data/Process_real_data.mat','Proc_data','MA_matrix')

filepath = 'Processed_data/Real_NN_8layers_act.mat';
load(filepath)% File exists.
Hb_NN = Y_real_act;


load('Processed_data/number_array.mat')

SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];

n_NN = zeros(size(MA_matrix,1),1);
for j = 1:length(number_array)
    if j == 1
        start_point = 1;
    else
        start_point = sum(number_array(1:(j-1))) + 1;
    end
    end_point = sum(number_array(1:j));
    
    HbO_NN = Hb_NN(start_point : end_point, 1:512);
    HbR_NN = Hb_NN(start_point : end_point, 513:end);
    for i = 1:size(HbO_NN,1)
        dc_HbO = HbO_NN(i,:);
        dc_HbR = HbR_NN(i,:);
        dc = [dc_HbO;dc_HbR]';
        dod = hmrConc2OD( dc, SD1, ppf );
        [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs_new, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
        n_MA = count_MA(tIncChAuto);
        n_NN(j) = n_NN(j) + n_MA;
    end
end

MA_matrix = [MA_matrix, n_NN];

figure
errorbar(1:length(number_array),mean(MA_matrix(1:end-1,:),1),std(MA_matrix(1:end-1,:),[],1), 'b' )
hold on
plot(1:length(number_array),MA_matrix(end,:),'ro-')

ylabel('Residual motion artifacts')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
% ylim([0 40])
xlim([0.5 8.5])

set(gcf,'Position',[3   431   390   274]);
box off

%% Figure 3C: Act vs. No act.

% test_file = Proc_data(end);
Real_matrix = zeros(14, 8);
Real_matrix_act = zeros(14, 8);

Real_matrix(:,1) = squeeze(mean(Proc_data(end).dc_no_crct(1:512,1,:),1));
Real_matrix(:,2) = squeeze(mean(Proc_data(end).dc_Spline(1:512,1,:),1));
Real_matrix(:,3) = squeeze(mean(Proc_data(end).dc_Wavelet(1:512,1,:),1));
Real_matrix(:,4) = squeeze(mean(Proc_data(end).dc_Kalman(1:512,1,:),1));
Real_matrix(:,5) = squeeze(mean(Proc_data(end).dc_PCA99(1:512,1,:),1));
Real_matrix(:,6) = squeeze(mean(Proc_data(end).dc_PCA97(1:512,1,:),1));
Real_matrix(:,7) = squeeze(mean(Proc_data(end).dc_Cbsi(1:512,1,:),1));

Real_matrix_act(:,1) = squeeze(mean(Proc_data(end).dc_act_no_crct(1:512,1,:),1));
Real_matrix_act(:,2) = squeeze(mean(Proc_data(end).dc_act_Spline(1:512,1,:),1));
Real_matrix_act(:,3) = squeeze(mean(Proc_data(end).dc_act_Wavelet(1:512,1,:),1));
Real_matrix_act(:,4) = squeeze(mean(Proc_data(end).dc_act_Kalman(1:512,1,:),1));
Real_matrix_act(:,5) = squeeze(mean(Proc_data(end).dc_act_PCA99(1:512,1,:),1));
Real_matrix_act(:,6) = squeeze(mean(Proc_data(end).dc_act_PCA97(1:512,1,:),1));
Real_matrix_act(:,7) = squeeze(mean(Proc_data(end).dc_act_Cbsi(1:512,1,:),1));

filepath = 'Processed_data/Real_NN_8layers.mat';
load(filepath)% File exists.
Hb_NN = Y_real;

j = length(number_array);
start_point = sum(number_array(1:(j-1))) + 1;
end_point = sum(number_array(1:j));

HbO_NN = mean(Hb_NN(start_point : end_point, 1:512),2);
HbO_NN = reshape(HbO_NN,[5,14]);
Real_matrix(:,8) = mean(HbO_NN,1)';

filepath = 'Processed_data/Real_NN_8layers_act.mat';
load(filepath)% File exists.
Hb_NN = Y_real_act;

j = length(number_array);
start_point = sum(number_array(1:(j-1))) + 1;
end_point = sum(number_array(1:j));

HbO_NN = mean(Hb_NN(start_point : end_point, 1:512),2);
HbO_NN = reshape(HbO_NN,[5,14]);
Real_matrix_act(:,8) = mean(HbO_NN,1)';

figure
boxplot(Real_matrix.*1e6, 'positions', (1:length(number_array)) - 0.2, 'colors','b', 'widths', 0.3, 'outliersize', 2, 'symbol', 'b.');
hold on
boxplot(Real_matrix_act.*1e6, 'positions', (1:length(number_array)) + 0.2, 'colors','r', 'widths', 0.3, 'outliersize', 2, 'symbol', 'r.');
hold on
plot([0 9],[0 0],'k-')
ylabel('Sum \Delta HbO (\muMol)')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
xlim([0.5 8.5])

set(gcf,'Position',[3   431   390   274]);
box off


% t test

for i = 1:size(Real_matrix,2)
    x1 = Real_matrix(:,i);
    x2 = Real_matrix_act(:,i);
    [~,p] = ttest2(x1, x2);
    fprintf('For %s: p = %.3f\n', labels{i},p)
end

%%  Figure 3D-I: Example
clear all
load('Processed_data/Testing_Spline.mat')
load('Processed_data/Testing_Wavelet01.mat')
load('Processed_data/Testing_Kalman.mat')
load('Processed_data/Testing_PCA99.mat')
load('Processed_data/Testing_PCA97.mat')
load('Processed_data/Testing_Cbsi.mat')


load('Processed_data/SimulateData.mat','HRF_test_noised','HRF_test')

[m,n] = size(HRF_test_noised);
HbO_test_noised = HRF_test_noised(1:m/2,:);
HbO_test = HRF_test(1:m/2,:);
HbO_no_crct = zeros(size(HbO_Cbsi));
HbO_real = zeros(size(HbO_Cbsi));

for i = 1:size(HbO_test,1)
    HbO_no_crct(i,:) = mean(reshape(HbO_test_noised(i,:),512,5),2);
    HbO_real(i,:) = mean(reshape(HbO_test(i,:),512,5),2);
end

load('Processed_data/Test_NN_8layers.mat')


figure('Renderer', 'painters','Position',[72        1229         989         196]);
subplot(2,3,1)
piece = 1;
plot(HbO_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbO_Spline(piece,:)*1e6,'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
plot(HbO_Wavelet01(piece,:)*1e6,'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
plot(HbO_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
plot(HbO_PCA99(piece,:)*1e6,'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
plot(HbO_PCA97(piece,:)*1e6,'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
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
