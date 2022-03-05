fold_folder = 'leave_7_out';
load(fullfile('Processed_data', fold_folder, 'Testing_Spline.mat'))
load(fullfile('Processed_data', fold_folder, 'Testing_Wavelet01.mat'))
load(fullfile('Processed_data', fold_folder, 'Testing_Kalman.mat'))
load(fullfile('Processed_data', fold_folder, 'Testing_Cbsi.mat'))

load(fullfile('Processed_data', fold_folder, 'SimulateData.mat'),'HRF_test_noised','HRF_test')

[m,n] = size(HRF_test_noised);
HbO_test_noised = HRF_test_noised(1:m/2,:);
HbO_test = HRF_test(1:m/2,:);
HbR_test_noised = HRF_test_noised(m/2+1:end,:);
HbR_test = HRF_test(m/2+1:end,:);
HbO_no_crct = zeros(size(HbO_Cbsi));
HbR_no_crct = zeros(size(HbO_Cbsi));
HbO_real = zeros(size(HbO_Cbsi));
HbR_real = zeros(size(HbO_Cbsi));

t  = 1/fs_new:1/fs_new:size(HbO_test_noised,2)/fs_new;
s  = zeros(1,length(t));
s((rt):512:length(t)) = 1;
tIncMan=ones(size(t))';

for i = 1:size(HbO_test,1)
    dc_HbO  = HbO_test_noised(i,:);
    dc_HbR  = HbR_test_noised(i,:);
    dc      =   [dc_HbO;dc_HbR]';
    [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
    HbO_no_crct(i,:) = dc_avg(:,1)';
    HbR_no_crct(i,:) = dc_avg(:,2)';
    
    HbO_real(i,:) = HbO_test(i,1:512);
    HbR_real(i,:) = HbR_test(i,1:512);
end

load(fullfile('Processed_data', fold_folder, 'Test_NN_8layers.mat'))
HbO_NN = zeros(size(HbO_Cbsi));
HbR_NN = zeros(size(HbO_Cbsi));
j = 1;
for i = 1:5:size(Y_test,1)
    HbO_NN(j,:) = mean(Y_test(i:i+4,1:512),1);
    HbR_NN(j,:) = mean(Y_test(i:i+4,513:end),1);
    j = j + 1;
end
%%
f1 = figure;
subplot(2,3,1)
% piece = 3;
% piece = 4;
piece = 2;
h1 = plot3(1*ones(512),1:512, HbO_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','Real','LineWidth',3);hold on;
h2 = plot3(2*ones(512),1:512, HbO_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
h3 = plot3(3*ones(512),1:512, HbO_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
h4 = plot3(4*ones(512),1:512, HbO_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
h5 = plot3(5*ones(512),1:512, HbO_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
h6 = plot3(6*ones(512),1:512, HbO_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
h7 = plot3(7*ones(512),1:512, HbO_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbO (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new)
hold off
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
% legend([h1; h2],{'Real','No correction'})
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)

subplot(2,3,4)
plot3(1*ones(512),1:512,HbR_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot3(2*ones(512),1:512,HbR_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot3(3*ones(512),1:512,HbR_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
plot3(4*ones(512),1:512,HbR_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
plot3(5*ones(512),1:512,HbR_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
plot3(6*ones(512),1:512,HbR_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot3(7*ones(512),1:512,HbR_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbR (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new) 
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)

% % show a bigger figure to the reviewers
f2 = figure;
subplot(231)
plot(1:512, HbO_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbO_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbO_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbO (\muMol\cdotmm)')
set(gca,'fontsize',10)

subplot(234)
plot(1:512, HbR_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbR_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbR_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbR (\muMol\cdotmm)')

figure(f1);
%%
% real no act

load('Processed_data/Process_real_data.mat','Proc_data')
file_number = 8;

HbO_real = zeros(512,14)';
HbO_no_crct = squeeze(Proc_data(file_number).dc_no_crct(1:512,1,:))';
HbO_Spline = squeeze(Proc_data(file_number).dc_Spline(1:512,1,:))';
HbO_Wavelet01 = squeeze(Proc_data(file_number).dc_Wavelet(1:512,1,:))';
HbO_Kalman = squeeze(Proc_data(file_number).dc_Kalman(1:512,1,:))';
HbO_PCA97 = squeeze(Proc_data(file_number).dc_PCA97(1:512,1,:))';
HbO_Cbsi = squeeze(Proc_data(file_number).dc_Cbsi(1:512,1,:))';

HbR_real = zeros(512,14)';
HbR_no_crct = squeeze(Proc_data(file_number).dc_no_crct(1:512,2,:))';
HbR_Spline = squeeze(Proc_data(file_number).dc_Spline(1:512,2,:))';
HbR_Wavelet01 = squeeze(Proc_data(file_number).dc_Wavelet(1:512,2,:))';
HbR_Kalman = squeeze(Proc_data(file_number).dc_Kalman(1:512,2,:))';
HbR_PCA97 = squeeze(Proc_data(file_number).dc_PCA97(1:512,2,:))';
HbR_Cbsi = squeeze(Proc_data(file_number).dc_Cbsi(1:512,2,:))';

DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
for subfolder = 1:length(subfolders)
%     fprintf('subfolder is %d\n', subfolder)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Real_NN_8layers.mat');
    % load data
    load(filepath)
    Hb_NN = Y_real;
    
    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    HbO_NN = reshape(HbO_NN',[],14)';
    HbR_NN = reshape(HbR_NN',[],14)';

    dc_HbO = HbO_NN;
    dc_HbR = HbR_NN;
    dc = [dc_HbO;dc_HbR]';
    [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
    Proc_data(subfolder).dc_NN = dc_avg';
end

HbO_NN = squeeze(Proc_data(file_number).dc_NN(1:14,:));
HbR_NN = squeeze(Proc_data(file_number).dc_NN(15:end,:));
%%
subplot(2,3,2)
% piece = 4;
piece = 2;
plot3(1*ones(512),1:512, HbO_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',3);hold on;
plot3(2*ones(512),1:512, HbO_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot3(3*ones(512),1:512, HbO_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
plot3(4*ones(512),1:512, HbO_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
plot3(5*ones(512),1:512, HbO_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
plot3(6*ones(512),1:512, HbO_PCA97(piece,:)*1e6,'color','c','linestyle','-','DisplayName','PCA','LineWidth',1.5);hold on;
plot3(7*ones(512),1:512, HbO_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot3(8*ones(512),1:512, HbO_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbO (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new)
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)

subplot(2,3,5)
plot3(1*ones(512),1:512,HbR_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot3(2*ones(512),1:512,HbR_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot3(3*ones(512),1:512,HbR_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
plot3(4*ones(512),1:512,HbR_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
plot3(5*ones(512),1:512,HbR_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
plot3(6*ones(512),1:512,HbR_PCA97(piece,:)*1e6,'color','c','linestyle','-','DisplayName','PCA','LineWidth',1.5);hold on;
plot3(7*ones(512),1:512,HbR_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot3(8*ones(512),1:512,HbR_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbR (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new) 
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)

figure(f2)
% figure
subplot(232)
plot(1:512, HbO_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbO_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbO_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbO (\muMol\cdotmm)')
set(gca,'fontsize',10)
ylim([-150 150])

subplot(235)
plot(1:512, HbR_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbR_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbR_NN(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbR (\muMol\cdotmm)')
set(gca,'fontsize',10)
ylim([-150 150])

figure(f1);
%% with act

% load('Processed_data/Process_real_data.mat','Proc_data','MA_matrix')
% load('Processed_data/RealData.mat','net_input')
HbO_real = repmat(Proc_data(file_number).HRF.HbO(1:512),14,1);
HbO_no_crct = squeeze(Proc_data(file_number).dc_act_no_crct(1:512,1,:))';
HbO_Spline = squeeze(Proc_data(file_number).dc_act_Spline(1:512,1,:))';
HbO_Wavelet01 = squeeze(Proc_data(file_number).dc_act_Wavelet(1:512,1,:))';
HbO_Kalman = squeeze(Proc_data(file_number).dc_act_Kalman(1:512,1,:))';
HbO_PCA97 = squeeze(Proc_data(file_number).dc_act_PCA97(1:512,1,:))';
HbO_Cbsi = squeeze(Proc_data(file_number).dc_act_Cbsi(1:512,1,:))';

HbR_real = repmat(Proc_data(file_number).HRF.HbR(1:512),14,1);
HbR_no_crct = squeeze(Proc_data(file_number).dc_act_no_crct(1:512,2,:))';
HbR_Spline = squeeze(Proc_data(file_number).dc_act_Spline(1:512,2,:))';
HbR_Wavelet01 = squeeze(Proc_data(file_number).dc_act_Wavelet(1:512,2,:))';
HbR_Kalman = squeeze(Proc_data(file_number).dc_act_Kalman(1:512,2,:))';
HbR_PCA97 = squeeze(Proc_data(file_number).dc_act_PCA97(1:512,2,:))';
HbR_Cbsi = squeeze(Proc_data(file_number).dc_act_Cbsi(1:512,2,:))';

DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
for subfolder = 1:length(subfolders)
%     fprintf('subfolder is %d\n', subfolder)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Real_NN_8layers_act.mat');
    load(filepath)
    Hb_NN = Y_real_act;
    
    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    HbO_NN = reshape(HbO_NN',[],14)';
    HbR_NN = reshape(HbR_NN',[],14)';

    dc_HbO = HbO_NN;
    dc_HbR = HbR_NN;
    dc = [dc_HbO;dc_HbR]';
    [dc_avg, ~, ~, ~, ~, ~] = hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] );
    Proc_data(subfolder).dc_act_NN = dc_avg';
end

HbO_NN_act = squeeze(Proc_data(file_number).dc_act_NN(1:14,:));
HbR_NN_act = squeeze(Proc_data(file_number).dc_act_NN(15:end,:));
%%

subplot(2,3,3)
% figure
% piece = 1;
plot3(1*ones(512),1:512, HbO_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',3);hold on;
plot3(2*ones(512),1:512, HbO_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot3(3*ones(512),1:512, HbO_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
plot3(4*ones(512),1:512, HbO_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
plot3(5*ones(512),1:512, HbO_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
plot3(6*ones(512),1:512, HbO_PCA97(piece,:)*1e6,'color','c','linestyle','-','DisplayName','PCA','LineWidth',1.5);hold on;
plot3(7*ones(512),1:512, HbO_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot3(8*ones(512),1:512, HbO_NN_act(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbO (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new)
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
hold off
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)

% figure
subplot(2,3,6)
plot3(1*ones(512),1:512,HbR_real(piece,:)*1e6,'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot3(2*ones(512),1:512,HbR_no_crct(piece,:)*1e6,'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot3(3*ones(512),1:512,HbR_Spline(piece,:)*1e6,'color','b','linestyle','-','DisplayName','Spline','LineWidth',1.5);hold on;
plot3(4*ones(512),1:512,HbR_Wavelet01(piece,:)*1e6,'color','g','linestyle','-','DisplayName','Wavelet','LineWidth',1.5);hold on;
plot3(5*ones(512),1:512,HbR_Kalman(piece,:)*1e6,'color',[0.9290 0.6940 0.1250],'linestyle','-','DisplayName','Kalman','LineWidth',1.5);hold on;
plot3(6*ones(512),1:512,HbR_PCA97(piece,:)*1e6,'color','c','linestyle','-','DisplayName','PCA','LineWidth',1.5);hold on;
plot3(7*ones(512),1:512,HbR_Cbsi(piece,:)*1e6,'Color',[153, 102, 51]./255,'linestyle','-','DisplayName','Cbsi','LineWidth',1.5);hold on;
plot3(8*ones(512),1:512,HbR_NN_act(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5);hold on;
set(gca,'Ydir','reverse')
zlabel('\Delta HbR (\muMol\cdotmm)')
yticks([0 20 40 60]*fs_new) 
yticklabels({'0s','20s','40s','60s'})
ylim([1 512])
hold off
% legend('Location','northeastoutside')
% legend show
set(gca,'fontsize',10)
set(gcf,'position',[35   436   960   543])

% show a bigger figure to the reviewers
figure(f2)
subplot(233)
plot(1:512, HbO_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbO_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbO_NN_act(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbO (\muMol\cdotmm)')
set(gca,'fontsize',10)
ylim([-150 150])

subplot(236)
plot(1:512, HbR_real(piece,:)*1e6, 'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(1:512, HbR_no_crct(piece,:)*1e6, 'color',[105, 105, 105]./255,'linestyle','-','DisplayName','No correction','LineWidth',2);hold on;
plot(1:512, HbR_NN_act(piece,:)*1e6,'color','r','linestyle','-','DisplayName','DAE','LineWidth',1.5)
xticks([0 20 40 60]*fs_new) 
xticklabels({'0s','20s','40s','60s'})
xlim([1 512])
ylabel('\Delta HbR (\muMol\cdotmm)')
set(gca,'fontsize',10)
ylim([-150 150])

set(gcf,'position',[35   436   960   543])