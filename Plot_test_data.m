close all
%% add homer path
pathHomer   =   '../../Tools/homer2_src_v2_3_10202017';
oldpath     =   cd(pathHomer);
setpaths;
cd(oldpath);
%% load data
load('Processed_data/HRF_test.mat','HRF_test')
load('Processed_data/HRF_test_noised.mat','HRF_test_noised')

[m,n]       =   size(HRF_test_noised);
HbO_noised  =   HRF_test_noised(1:m/2,:);
HbR_noised  =   HRF_test_noised(m/2+1:end,:);
HbO         =   HRF_test(1:m/2,:);
HbR         =   HRF_test(m/2+1:end,:);

load('Processed_data/Testing_Spline.mat')
load('Processed_data/Testing_Wavelet05.mat')
load('Processed_data/Testing_Wavelet35.mat')
load('Processed_data/Testing_Kalman.mat')
load('Processed_data/Testing_PCA99.mat')
load('Processed_data/Testing_PCA50.mat')
load('Processed_data/Testing_Cbsi.mat')
load('Processed_data/Testing_NN.mat')

filepath = 'Processed_data/Test_NN_8layers.mat';

if exist(filepath, 'file') == 2
    load(filepath)% File exists.
    Hb_NN = Y_test;
else
    Hb_NN = zeros(size(HbO));
    % File does not exist.
end

HbO_NN = Hb_NN(:,1:size(HbO,2));
HbR_NN = Hb_NN(:,size(HbO,2)+1:end);

HbO             =   HbO * 1e6;
HbR             =   HbR * 1e6;
HbO_NN          =   HbO_NN * 1e6;
HbR_NN          =   HbR_NN * 1e6;
HbO_noised      =   HbO_noised *1e6;
HbR_noised      =   HbR_noised *1e6;

HbO_Spline      =   HbO_Spline * 1e6;
HbO_Wavelet05   =   HbO_Wavelet05 * 1e6;
HbO_Wavelet35   =   HbO_Wavelet35 * 1e6;
HbO_Kalman      =   HbO_Kalman * 1e6;
HbO_PCA99       =   HbO_PCA99 * 1e6;
HbO_PCA50       =   HbO_PCA50 * 1e6;
HbO_Cbsi        =   HbO_Cbsi * 1e6;

HbR_Spline      =   HbR_Spline * 1e6;
HbR_Wavelet05   =   HbR_Wavelet05 * 1e6;
HbR_Wavelet35   =   HbR_Wavelet35 * 1e6;
HbR_Kalman      =   HbR_Kalman * 1e6;
HbR_PCA99       =   HbR_PCA99 * 1e6;
HbR_PCA50       =   HbR_PCA50 * 1e6;
HbR_Cbsi        =   HbR_Cbsi * 1e6;

labels = {'No correction','Spline','Wavelet05','Wavelet35','Kalman','PCA99','PCA50','Cbsi','DAE'};
%% figure 0: Example plot
piece = input('Which piece you want to show? ');
% 3 and 5

figure('Renderer', 'painters','Position',[72        1229         989         196]);
subplot(1,2,1)
plot(HbO(piece,:),'color',[128 128 128]./255,'linestyle','-','DisplayName','Ground truth','LineWidth',2.5);hold on;
plot(HbO_noised(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbO_Spline(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1);hold on;
plot(HbO_Wavelet05(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1);hold on;
plot(HbO_Wavelet35(piece,:),'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1);hold on;
plot(HbO_Kalman(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1);hold on;
plot(HbO_PCA99(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1);hold on;
plot(HbO_PCA50(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1);hold on;
plot(HbO_Cbsi(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1);hold on;
plot(HbO_NN(piece,:),'color','r','linestyle','-','DisplayName','DAE','LineWidth',2.5);hold on;
ylabel('\Delta HbO (\muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs)
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)

subplot(1,2,2)
plot(HbR(piece,:),'color',[128 128 128]./255,'linestyle','-','DisplayName','Ground truth','LineWidth',2.5);hold on;
plot(HbR_noised(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
plot(HbR_Spline(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1);hold on;
plot(HbR_Wavelet05(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1);hold on;
plot(HbR_Wavelet35(piece,:),'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1);hold on;
plot(HbR_Kalman(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1);hold on;
plot(HbR_PCA99(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1);hold on;
plot(HbR_PCA50(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1);hold on;
plot(HbR_Cbsi(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1);hold on;
plot(HbR_NN(piece,:),'color','r','linestyle','-','DisplayName','DAE','LineWidth',2);hold on;
ylabel('\Delta HbR (\muMol)')
fs = 7.8125;
xticks([0 20 40 60]*fs) 
xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
set(gca,'fontsize',12)

% subplot(2,2,3)
% % plot(HbO(piece,:),'color',[128 128 128]./255,'linestyle','-','DisplayName','Ground truth','LineWidth',2.5);hold on;
% plot(HbO_noised(piece,:) - HbO(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
% plot(HbO_Spline(piece,:) - HbO(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
% plot(HbO_Wavelet05(piece,:) - HbO(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
% plot(HbO_Wavelet35(piece,:) - HbO(piece,:),'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1.5);hold on;
% plot(HbO_Kalman(piece,:) - HbO(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
% plot(HbO_PCA99(piece,:) - HbO(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
% plot(HbO_PCA50(piece,:) - HbO(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
% plot(HbO_Cbsi(piece,:) - HbO(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
% plot(HbO_NN(piece,:) - HbO(piece,:),'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
% ylabel('\Delta HbO (\muMol)')
% fs = 7.8125;
% xticks([0 20 40 60]*fs)
% xticklabels({'0s','20s','40s','60s'})
% legend('Location','northeastoutside')
% legend show
% set(gca,'fontsize',12)
% mse_noised = mean((HbO_noised(piece,:) - HbO(piece,:)).^2);
% mse_Spline = mean((HbO_Spline(piece,:) - HbO(piece,:)).^2);
% mse_Wavelet05 = mean((HbO_Wavelet05(piece,:) - HbO(piece,:)).^2);
% mse_Wavelet35 = mean((HbO_Wavelet35(piece,:) - HbO(piece,:)).^2);
% mse_Kalman = mean((HbO_Kalman(piece,:) - HbO(piece,:)).^2);
% mse_PCA99 = mean((HbO_PCA99(piece,:) - HbO(piece,:)).^2);
% mse_PCA50 = mean((HbO_PCA50(piece,:) - HbO(piece,:)).^2);
% mse_Cbsi = mean((HbO_Cbsi(piece,:) - HbO(piece,:)).^2);
% mse_NN = mean((HbO_NN(piece,:) - HbO(piece,:)).^2);
% 
% 
% fprintf('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n',mse_noised,mse_Spline,mse_Wavelet05,mse_Wavelet35,mse_Kalman,...
%     mse_PCA99,mse_PCA50,mse_Cbsi,mse_NN)
% 
% subplot(2,2,4)
% % plot(HbR(piece,:),'color',[128 128 128]./255,'linestyle','-','DisplayName','Ground truth','LineWidth',2.5);hold on;
% plot(HbR_noised(piece,:)-HbR(piece,:),'color','k','linestyle','-','DisplayName','No correction','LineWidth',2.5);hold on;
% plot(HbR_Spline(piece,:) - HbR(piece,:),'color','b','linestyle','-.','DisplayName','Spline','LineWidth',1.5);hold on;
% plot(HbR_Wavelet05(piece,:) - HbR(piece,:),'color','g','linestyle','-.','DisplayName','Wavelet05','LineWidth',1.5);hold on;
% plot(HbR_Wavelet35(piece,:) - HbR(piece,:),'color','g','linestyle',':','DisplayName','Wavelet35','LineWidth',1.5);hold on;
% plot(HbR_Kalman(piece,:) - HbR(piece,:),'color',[0.9290 0.6940 0.1250],'linestyle','-.','DisplayName','Kalman','LineWidth',1.5);hold on;
% plot(HbR_PCA99(piece,:) - HbR(piece,:),'color','m','linestyle','-.','DisplayName','PCA99','LineWidth',1.5);hold on;
% plot(HbR_PCA50(piece,:) - HbR(piece,:),'color','c','linestyle','-.','DisplayName','PCA50','LineWidth',1.5);hold on;
% plot(HbR_Cbsi(piece,:) - HbR(piece,:),'Color',[153, 102, 51]./255,'linestyle','-.','DisplayName','Cbsi','LineWidth',1.5);hold on;
% plot(HbR_NN(piece,:) - HbR(piece,:),'color','r','linestyle','-.','DisplayName','DAE','LineWidth',1.5);hold on;
% ylabel('\Delta HbR (\muMol)')
% fs = 7.8125;
% xticks([0 20 40 60]*fs) 
% xticklabels({'0s','20s','40s','60s'})
% legend('Location','northeastoutside')
% legend show
% set(gca,'fontsize',12)
return
%% 1. bar plot of n
% number of noise
load('Processed_data/peak_n_list.mat','peak_n_list')
load('Processed_data/shift_n_list.mat','shift_n_list')
load('Processed_data/random_profile.mat','p')

m           =   size(peak_n_list,1);
n_train     =   round((m/2)*0.8);
n_val       =   round((m/2)*0.1);
test_idx    =   [p(n_train+n_val+1:end),p(n_train+n_val+1:end)+m/2];

peak_n_list_test    =   peak_n_list(test_idx,:);
shift_n_list_test   =   shift_n_list(test_idx,:);
n_test              =   size(peak_n_list_test,1)/2;
n_MA_total_HbO      =   sum(peak_n_list_test(1:n_test,:))+sum(shift_n_list_test(1:n_test,:));
n_MA_total_HbR      =   sum(peak_n_list_test(n_test+1:end,:))+sum(shift_n_list_test(n_test+1:end,:));

MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,...
    n_Wavelet35_HbO,n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,...
    n_Wavelet35_HbR,n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];

MA_list_new = MA_list(1,:);
MA_list_percetage = MA_list_new./MA_list_new(1);

figure
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
%% calculate of mse
MSE_HbO         =   zeros(9,size(HbO,1));
MSE_HbO(1,:)    =   mean((HbO-HbO_noised).^2,2);
MSE_HbO(2,:)    =   mean((HbO-HbO_Spline).^2,2);       % Spline
MSE_HbO(3,:)    =   mean((HbO-HbO_Wavelet05).^2,2);    % Wavelet05
MSE_HbO(4,:)    =   mean((HbO-HbO_Wavelet35).^2,2);    % Wavelet35
MSE_HbO(5,:)    =   mean((HbO-HbO_Kalman).^2,2);       % Kalman
MSE_HbO(6,:)    =   mean((HbO-HbO_PCA99).^2,2);        % PCA99
MSE_HbO(7,:)    =   mean((HbO-HbO_PCA50).^2,2);        % PCA50
MSE_HbO(8,:)    =   mean((HbO-HbO_Cbsi).^2,2);         % Cbsi
MSE_HbO(9,:)    =   mean((HbO-HbO_NN).^2,2);           % NN

MSE_HbR         =   zeros(9,size(HbR,1));
MSE_HbR(1,:)    =   mean((HbR-HbR_noised).^2,2);
MSE_HbR(2,:)    =   mean((HbR-HbR_Spline).^2,2);       % Spline
MSE_HbR(3,:)    =   mean((HbR-HbR_Wavelet05).^2,2);    % Wavelet05
MSE_HbR(4,:)    =   mean((HbR-HbR_Wavelet35).^2,2);    % Wavelet35
MSE_HbR(5,:)    =   mean((HbR-HbR_Kalman).^2,2);       % Kalman
MSE_HbR(6,:)    =   mean((HbR-HbR_PCA99).^2,2);        % PCA99
MSE_HbR(7,:)    =   mean((HbR-HbR_PCA50).^2,2);        % PCA50
MSE_HbR(8,:)    =   mean((HbR-HbR_Cbsi).^2,2);         % Cbsi
MSE_HbR(9,:)    =   mean((HbR-HbR_NN).^2,2);           % NN

mean_MSE_HbO = mean(MSE_HbO,2);
std_MSE_HbO = std(MSE_HbO,0,2);
fprintf('HbO MSE:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_MSE_HbO(i),std_MSE_HbO(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(MSE_HbO(1,:),MSE_HbO(i,:));
    fprintf('p = %.5f\n',p)
end
mean_MSE_HbR = mean(MSE_HbR,2);
std_MSE_HbR = std(MSE_HbR,0,2);
fprintf('HbR MSE:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_MSE_HbR(i),std_MSE_HbR(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(MSE_HbR(1,:),MSE_HbR(i,:));
    fprintf('p = %.5f\n',p)
end

%% calculate of cnr
cnr_HbO         =   zeros(9,size(HbO,1));
cnr_HbR         =   zeros(9,size(HbO,1));
cnr_HbO(1,:)    =   cnr_cal(HbO,HbO_noised,0);
cnr_HbO(2,:)    =   cnr_cal(HbO,HbO_Spline,0);
cnr_HbO(3,:)    =   cnr_cal(HbO,HbO_Wavelet05,0);
cnr_HbO(4,:)    =   cnr_cal(HbO,HbO_Wavelet35,0);
cnr_HbO(5,:)    =   cnr_cal(HbO,HbO_Kalman,0);
cnr_HbO(6,:)    =   cnr_cal(HbO,HbO_PCA99,0);
cnr_HbO(7,:)    =   cnr_cal(HbO,HbO_PCA50,0);
cnr_HbO(8,:)    =   cnr_cal(HbO,HbO_Cbsi,0);
cnr_HbO(9,:)    =   cnr_cal(HbO,HbO_NN,0);

cnr_HbR(1,:)    =   cnr_cal(HbR,HbR_noised,1);
cnr_HbR(2,:)    =   cnr_cal(HbR,HbR_Spline,1);
cnr_HbR(3,:)    =   cnr_cal(HbR,HbR_Wavelet05,1);
cnr_HbR(4,:)    =   cnr_cal(HbR,HbR_Wavelet35,1);
cnr_HbR(5,:)    =   cnr_cal(HbR,HbR_Kalman,1);
cnr_HbR(6,:)    =   cnr_cal(HbR,HbR_PCA99,1);
cnr_HbR(7,:)    =   cnr_cal(HbR,HbR_PCA50,1);
cnr_HbR(8,:)    =   cnr_cal(HbR,HbR_Cbsi,1);
cnr_HbR(9,:)    =   cnr_cal(HbR,HbR_NN,1);

mean_CNR_HbO = mean(cnr_HbO,2);
std_CNR_HbO = std(cnr_HbO,0,2);
fprintf('HbO CNR:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_CNR_HbO(i),std_CNR_HbO(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(cnr_HbO(1,:),cnr_HbO(i,:));
    fprintf('p = %.5f\n',p)
end
mean_CNR_HbR = mean(cnr_HbR,2);
std_CNR_HbR = std(cnr_HbR,0,2);
fprintf('HbR CNR:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_CNR_HbR(i),std_CNR_HbR(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(cnr_HbR(1,:),cnr_HbR(i,:));
    fprintf('p = %.5f\n',p)
end
return
%% which AUC range is linear with MSE
fs = 7.8125;
MSE_value = MSE_HbO(1,:);
c_c = zeros(10,1);
j = 1;
for i = 1:1:10
    noised_data = HbO_noised - repmat(mean(HbO_noised(:,1:round(fs*1)),2),1,512);
    AUC = abs(trapz(noised_data(:,1:round(fs*i))./fs,2));
    corrcoef_value = corrcoef(MSE_value',AUC);
    c_c(j) = corrcoef_value(1,2);
    j = j + 1;
end

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(1:j-1,c_c,'Linewidth',1,'Marker','o','markerfacecolor','b')
% ylabel('Correlation Coefficient')
% xlabel('AUC time range (s)')
% title('Correlation of MSE and AUC')
set(gca,'FontName','Arial','FontSize',14)
box off
% xlim([1 10])
saveas(gcf,'Figures/AUC_examine','fig')
saveas(gcf,'Figures/AUC_examine','svg')
% 18 is the best
%% which AUC ratio range is linear with MSE
fs = 7.8125;
MSE_value = MSE_HbO(1,:);
AUC0_18 = abs(trapz(HbO_noised(:,1:round(fs*2))./fs,2));
c_c = zeros(10,1);
j = 1;
for i = 1:1:10
    nosed_data = HbO_noised - repmat(mean(HbO_noised(:,1:round(fs*1)),2),1,512);
    AUC_high = abs(trapz(nosed_data(:,round(fs*2):round(fs*(2+i)))./fs,2));
    AUC_ratio = AUC_high./AUC0_18;
    corrcoef_value = corrcoef(MSE_value,AUC_ratio);
    c_c(j) = corrcoef_value(2);
    j = j + 1;
end
figure('Renderer', 'painters', 'Position', [10 10 300 200])
h = plot((1:j-1)+2,c_c,'Linewidth',1,'Marker','o','markerfacecolor','b');
ax = ancestor(h, 'axes');
ax.YAxis.Exponent = 0;
% ylabel('Correlation Coefficient')
% xlabel('AUC time range to divide AUC0-18')
% title('Correlation of MSE and AUCratio')
set(gca,'FontName','Arial','FontSize',14)
box off
% xlim([2 20])
saveas(gcf,'Figures/AUCratio_examine','fig')
saveas(gcf,'Figures/AUCratio_examine','svg')

%% calculate of snr
snr_HbO         =   zeros(9,size(HbO,1));
snr_HbR         =   zeros(9,size(HbO,1));

snr_HbO(1,:)    =   snr_cal(HbO_noised);
snr_HbO(2,:)    =   snr_cal(HbO_Spline);
snr_HbO(3,:)    =   snr_cal(HbO_Wavelet05);
snr_HbO(4,:)    =   snr_cal(HbO_Wavelet35);
snr_HbO(5,:)    =   snr_cal(HbO_Kalman);
snr_HbO(6,:)    =   snr_cal(HbO_PCA99);
snr_HbO(7,:)    =   snr_cal(HbO_PCA50);
snr_HbO(8,:)    =   snr_cal(HbO_Cbsi);
snr_HbO(9,:)    =   snr_cal(HbO_NN);

snr_HbR(1,:)    =   snr_cal(HbR_noised);
snr_HbR(2,:)    =   snr_cal(HbR_Spline);
snr_HbR(3,:)    =   snr_cal(HbR_Wavelet05);
snr_HbR(4,:)    =   snr_cal(HbR_Wavelet35);
snr_HbR(5,:)    =   snr_cal(HbR_Kalman);
snr_HbR(6,:)    =   snr_cal(HbR_PCA99);
snr_HbR(7,:)    =   snr_cal(HbR_PCA50);
snr_HbR(8,:)    =   snr_cal(HbR_Cbsi);
snr_HbR(9,:)    =   snr_cal(HbR_NN);

mean_SNR_HbO = mean(snr_HbO,2);
std_SNR_HbO = std(snr_HbO,0,2);
fprintf('HbO SNR:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_SNR_HbO(i),std_SNR_HbO(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(snr_HbO(1,:),snr_HbR(i,:));
    fprintf('p = %.5f\n',p)
end
mean_SNR_HbR = mean(snr_HbR,2);
std_SNR_HbR = std(snr_HbR,0,2);
fprintf('HbR SNR:\n')
for i = 1:9
    fprintf('%.2f(%.2f)\t',mean_SNR_HbR(i),std_SNR_HbR(i))
    if i == 1
        fprintf('\n')
        continue
    end
    [h,p] = ttest(snr_HbR(1,:),snr_HbR(i,:));
    fprintf('p = %.5f\n',p)
end

% %% 2. plot the potion of trials with mse decreased
% % figure
% 
% x_data  =   MSE_HbO(1,:);
% y_data  =   MSE_HbO(2:end,:);
% 
% for i = 2:9
%     [h,p] = ttest(x_data,MSE_HbO(i,:));
%     fprintf('p = %.5f\n',p)
% end
% 
% delta_MSE_HbO = repmat(x_data,size(y_data,1),1) - y_data;
% mean_delta_MSE_HbO = mean(delta_MSE_HbO,2);
% n2      =   sum(y_data < repmat(x_data,size(y_data,1),1),2);
% n       =   size(x_data,2);
% 
% y = n2./n*100;
% b = bar(y,'facecolor',[108, 171, 215]./256);
% title('Proportion of trials whose MSE decreased')
% set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
% ytickformat(gca, 'percentage');
% set(gca, 'XTick', 1:size(MA_list,2)-1)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels(2:end))
% set(gca, 'Fontsize', 12)
% xtickangle(90)
% xlim([0 9])
% ylim([0 110])
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% for i = 1:length(b(1).YData)
%     labels1(i) = sprintf('%.0f%%',b(1).YData(i));
% end
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% set(gcf,'Position',[334   490   330   215]);
% ax              =   gca;
% ax.XGrid        =   'off';
% ax.YGrid        =   'on';
% b.EdgeColor     =   [1 1 1];
% b.FaceColor     =   'flat';
% b.CData(8,:)    =   [200, 14, 80]./255;
% box off
% % HbR
% figure
% 
% x_data  =   MSE_HbR(1,:);
% y_data  =   MSE_HbR(2:end,:);
% n2      =   sum(y_data<repmat(x_data,size(y_data,1),1),2);
% delta_MSE_HbR = repmat(x_data,size(y_data,1),1) - y_data;
% mean_delta_MSE_HbR = mean(delta_MSE_HbR,2);
% n       =   size(x_data,2);
% 
% y = n2./n*100;
% b = bar(y,'facecolor',[108, 171, 215]./256);
% title('Proportion of trials whose MSE decreased')
% set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
% ytickformat(gca, 'percentage');
% % gca.YGrid = 'on';
% set(gca, 'XTick', 1:size(MA_list,2)-1)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels(2:end))
% set(gca, 'Fontsize', 12)
% xtickangle(90)
% xlim([0 9])
% ylim([0 110])
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% 
% for i = 1:length(b(1).YData)
%     labels1(i) = sprintf('%.0f%%',b(1).YData(i));
% end
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% set(gcf,'Position',[665   490   330   215]);
% ax              =   gca;
% ax.XGrid        =   'off';
% ax.YGrid        =   'on';
% b.EdgeColor     =   [1 1 1];
% b.FaceColor     =   'flat';
% b.CData(8,:)    =   [200, 14, 80]./255;
% box off
% % table

% %% 3. plot the potion of trials with cnr decreased
% figure
% 
% x_data  =   cnr_HbO(1,:);
% y_data  =   cnr_HbO(2:end,:);
% n2      =   sum(y_data>repmat(x_data,size(y_data,1),1),2);
% n       =   size(x_data,2);
% 
% y = n2./n*100;
% b = bar(y,'facecolor',[108, 171, 215]./256);
% title('Proportion of trials whose CNR increased')
% set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
% ytickformat(gca, 'percentage');
% % gca.YGrid = 'on';
% set(gca, 'XTick', 1:size(MA_list,2)-1)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels(2:end))
% set(gca, 'Fontsize', 12)
% xtickangle(90)
% xlim([0 9])
% ylim([0 110])
% xtips1  = b(1).XEndPoints;
% ytips1  = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% 
% for i = 1:length(b(1).YData)
%     labels1(i) = sprintf('%.0f%%',b(1).YData(i));
% end
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% set(gcf,'Position',[996   490   330   215]);
% ax              =   gca;
% ax.XGrid        =   'off';
% ax.YGrid        =   'on';
% b.EdgeColor     =   [1 1 1];
% b.FaceColor     =   'flat';
% b.CData(8,:)    =   [200, 14, 80]./255;
% box off
% % HbR
% figure
% 
% x_data  =   cnr_HbR(1,:);
% y_data  =   cnr_HbR(2:end,:);
% 
% n2      =   sum(y_data<repmat(x_data,size(y_data,1),1),2);
% n       =   size(x_data,2);
% 
% y       =   n2./n*100;
% b       =   bar(y,'facecolor',[108, 171, 215]./256);
% title('Proportion of trials whose CNR increased')
% set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
% ytickformat(gca, 'percentage');
% 
% set(gca, 'XTick', 1:size(MA_list,2)-1)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels(2:end))
% set(gca, 'Fontsize', 12)
% xtickangle(90)
% xlim([0 9])
% ylim([0 110])
% xtips1  =   b(1).XEndPoints;
% ytips1  =   b(1).YEndPoints;
% labels1 =   string(b(1).YData);
% 
% for i = 1:length(b(1).YData)
%     labels1(i) = sprintf('%.0f%%',b(1).YData(i));
% end
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% set(gcf,'Position',[3   201   330   215]);
% ax              =   gca;
% ax.XGrid        =   'off';
% ax.YGrid        =   'on';
% b.EdgeColor     =   [1 1 1];
% b.FaceColor     =   'flat';
% b.CData(8,:)    =   [200, 14, 80]./255;
% box off
%% 4. low ns high noise vol.

peak_n_list_test    =   peak_n_list(test_idx,:);
shift_n_list_test   =   shift_n_list(test_idx,:);
m   =  size(peak_n_list_test);
peak_n_list_test_HbO    =   peak_n_list_test(1:m/2);
shift_n_list_test_HbO   =   shift_n_list_test(1:m/2);

noise_n         = peak_n_list_test_HbO + shift_n_list_test_HbO;
med_noise_n = median(noise_n);
low_noise_idx = find(noise_n<=med_noise_n);
high_noise_idx = find(noise_n>med_noise_n);
low_noise_mse = MSE_HbO(2:end,low_noise_idx);
high_noise_mse = MSE_HbO(2:end,high_noise_idx);
val = [mean(low_noise_mse,2),mean(high_noise_mse,2)]';
err  = [std(low_noise_mse,[],2),std(high_noise_mse,[],2)]'./2;

figure
hold on
bar(val')
legend('Low noise vol.','High noise vol.')
ylabel('(\muMol)^2')
set(gca, 'XTick', 1:size(MA_list,2)-1,'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:9))
xtickangle(90)
xlim([0.5 8.5])
set(gcf,'Position',[334   201   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
box off
hold off

%% 4. peak ns shift

peak_n_list_test = peak_n_list(test_idx,:);
shift_n_list_test = shift_n_list(test_idx,:);
m = size(peak_n_list_test);
peak_n_list_test_HbO = peak_n_list_test(1:m/2);
shift_n_list_test_HbO = shift_n_list_test(1:m/2);

noise_n = peak_n_list_test_HbO + shift_n_list_test_HbO;
med_noise_n = median(noise_n);
peak_idx = find(peak_n_list_test_HbO>shift_n_list_test_HbO);
shift_idx = find(peak_n_list_test_HbO<shift_n_list_test_HbO);
peak_mse = MSE_HbO(2:end,peak_idx);
shift_mse = MSE_HbO(2:end,shift_idx);
val = [mean(peak_mse,2),mean(shift_mse,2)]';
err  = [std(peak_mse,[],2),std(shift_mse,[],2)]'./2;

figure
hold on
bar(val')
legend('Peaks','Shifts')
ylabel('(\muMol)^2')
set(gca, 'XTick', 1:size(MA_list,2)-1,'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:8))
xtickangle(90)
xlim([0.5 8.5])
set(gcf,'Position',[665   203   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
box off
hold off
%% 4. High ns Low HRF amp.
load('Processed_data/amp_HRF_list.mat','amp_HRF_list')
load('Processed_data/random_profile.mat','p')

m = size(amp_HRF_list,1);
n_train = round((m/2)*0.8);
n_val = round((m/2)*0.1);
test_idx = [p(n_train+n_val+1:end),p(n_train+n_val+1:end)+m/2];

amp_HRF_list_test = amp_HRF_list(test_idx,:);
m = size(amp_HRF_list_test);
amp_HRF_list_test_HbO = amp_HRF_list_test(1:m/2);

med = median(amp_HRF_list_test_HbO);
high_idx = find(amp_HRF_list_test_HbO>med);
low_idx = find(amp_HRF_list_test_HbO<med);
high_mse = MSE_HbO(2:end,high_idx);
low_mse = MSE_HbO(2:end,low_idx);
val = [mean(high_mse,2),mean(low_mse,2)]';
err  = [std(high_mse,[],2),std(low_mse,[],2)]'./2;

figure
hold on
bar(val')
legend('High HRF amp.','Low HRF amp.')
ylabel('(\muMol)^2')
set(gca, 'XTick', 1:size(MA_list,2)-1,'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:8))
xtickangle(90)
xlim([0.5 8.5])
set(gcf,'Position',[997   204   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';

box off
hold off
%% 4. Early vs Late noise.
load('Processed_data/t0_peak_list.mat','t0_peak_list')
load('Processed_data/t0_shift_list.mat','t0_shift_list')
t0_peak_list_test = t0_peak_list(test_idx,:);
t0_shift_list_test = t0_shift_list(test_idx,:);
m = size(t0_peak_list_test);
t0_peak_list_test_HbO = t0_peak_list_test(1:m/2);
t0_shift_list_test_HbO = t0_shift_list_test(1:m/2);
med = 512/2;
early_idx_peak = find(t0_peak_list_test_HbO(peak_n_list_test_HbO~=0)<med);
early_idx_shift = find(t0_shift_list_test_HbO(shift_n_list_test_HbO~=0)<med);
early_idx = union(early_idx_peak,early_idx_shift);
late_idx_peak = find(t0_peak_list_test_HbO(peak_n_list_test_HbO~=0)>med);
late_idx_shift = find(t0_shift_list_test_HbO(shift_n_list_test_HbO~=0)>med);
late_idx = union(late_idx_peak,late_idx_shift);
early_mse = MSE_HbO(2:end,early_idx);
late_mse = MSE_HbO(2:end,late_idx);
val = [mean(early_mse,2),mean(late_mse,2)]';
err  = [std(early_mse,[],2),std(late_mse,[],2)]'./2;

figure
hold on
bar(val')
legend('Early noise','Late noise')
ylabel('(\muMol)^2')
set(gca, 'XTick', 1:size(MA_list,2)-1,'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:8))
xtickangle(90)
xlim([0.5 8.5])
set(gcf,'Position',[34   490   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
box off
hold off

%%
function cnr_mat = cnr_cal(Hb,Hb_noised,flag)
[m,n] = size(Hb);
fs = 7.8125;
cnr_mat = zeros(m,1);
for i = 1:m
    Hb_data = Hb(i,:);
    if flag == 0
        max_index_Hb = find(Hb_data == max(Hb_data));
    else
        max_index_Hb = find(Hb_data == min(Hb_data));
    end
    twosecond = max_index_Hb-round(1*fs):max_index_Hb+round(1*fs);
    cnr_mat(i) = abs(mean(Hb_noised(i,twosecond))-mean(Hb_noised(i,1:round(2*fs))))/std(Hb_noised(i,:)-Hb_data);
end
end

function snr_mat = snr_cal(Hb)

snr_mat = abs(mean(Hb,2)./std(Hb,0,2));

end