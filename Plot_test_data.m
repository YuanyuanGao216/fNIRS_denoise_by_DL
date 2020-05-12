clear all
close all
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

HbO = HbO * 1e6;
HbR = HbR * 1e6;
HbO_NN = HbO_NN * 1e6;
HbR_NN = HbR_NN * 1e6;
HbO_Spline = HbO_Spline * 1e6;
HbO_Wavelet05 = HbO_Wavelet05 * 1e6;
HbO_Wavelet35 = HbO_Wavelet35 * 1e6;
HbO_Kalman = HbO_Kalman * 1e6;
HbO_PCA99 = HbO_PCA99 * 1e6;
HbO_PCA50 = HbO_PCA50 * 1e6;
HbO_Cbsi = HbO_Cbsi * 1e6;

HbR_Spline = HbR_Spline * 1e6;
HbR_Wavelet05 = HbR_Wavelet05 * 1e6;
HbR_Wavelet35 = HbR_Wavelet35 * 1e6;
HbR_Kalman = HbR_Kalman * 1e6;
HbR_PCA99 = HbR_PCA99 * 1e6;
HbR_PCA50 = HbR_PCA50 * 1e6;
HbR_Cbsi = HbR_Cbsi * 1e6;

labels = {'No correction','Spline','Wavelet05','Wavelet35','Kalman','PCA99','PCA50','Cbsi','DAE'};

%% 1. bar plot of n
% number of noise
load('Processed_data/peak_n_list.mat','peak_n_list')
load('Processed_data/shift_n_list.mat','shift_n_list')
load('Processed_data/random_profile.mat','p')

m = size(peak_n_list,1);
n_train = round((m/2)*0.8);
n_val = round((m/2)*0.1);
test_idx = [p(n_train+n_val+1:end),p(n_train+n_val+1:end)+m/2];

peak_n_list_test = peak_n_list(test_idx,:);
shift_n_list_test = shift_n_list(test_idx,:);
n_test = size(peak_n_list_test,1)/2;
n_MA_total_HbO = sum(peak_n_list_test(1:n_test,:))+sum(shift_n_list_test(1:n_test,:));
n_MA_total_HbR = sum(peak_n_list_test(n_test+1:end,:))+sum(shift_n_list_test(n_test+1:end,:));

MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet35_HbO,n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet35_HbR,n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];


figure
b = bar(MA_list(1,:),'facecolor',[108, 171, 215]./256);
ylabel('No. of Motion Artifacts')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
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
%% calculate of mse
MSE_HbO = zeros(9,size(HbO,1));
MSE_HbO(1,:) = mean((HbO-HbO_noised).^2,2);
MSE_HbO(2,:) = mean((HbO-HbO_Spline).^2,2);       % Spline
MSE_HbO(3,:) = mean((HbO-HbO_Wavelet05).^2,2);    % Wavelet05
MSE_HbO(4,:) = mean((HbO-HbO_Wavelet35).^2,2);    % Wavelet35
MSE_HbO(5,:) = mean((HbO-HbO_Kalman).^2,2);       % Kalman
MSE_HbO(6,:) = mean((HbO-HbO_PCA99).^2,2);        % PCA99
MSE_HbO(7,:) = mean((HbO-HbO_PCA50).^2,2);        % PCA50
MSE_HbO(8,:) = mean((HbO-HbO_Cbsi).^2,2);         % Cbsi
MSE_HbO(9,:) = mean((HbO-HbO_NN).^2,2);           % NN

MSE_HbR = zeros(9,size(HbR,1));
MSE_HbR(1,:) = mean((HbR-HbR_noised).^2,2);
MSE_HbR(2,:) = mean((HbR-HbR_Spline).^2,2);       % Spline
MSE_HbR(3,:) = mean((HbR-HbR_Wavelet05).^2,2);    % Wavelet05
MSE_HbR(4,:) = mean((HbR-HbR_Wavelet35).^2,2);    % Wavelet35
MSE_HbR(5,:) = mean((HbR-HbR_Kalman).^2,2);       % Kalman
MSE_HbR(6,:) = mean((HbR-HbR_PCA99).^2,2);        % PCA99
MSE_HbR(7,:) = mean((HbR-HbR_PCA50).^2,2);        % PCA50
MSE_HbR(8,:) = mean((HbR-HbR_Cbsi).^2,2);         % Cbsi
MSE_HbR(9,:) = mean((HbR-HbR_NN).^2,2);           % NN
%% 2. plot the potion of trials with mse decreased
figure

x_data = MSE_HbO(1,:);
y_data = MSE_HbO(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y,'facecolor',[108, 171, 215]./256);
title('Proportion of trials whose MSE decreased')
set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
ytickformat(gca, 'percentage');
% gca.YGrid = 'on';
set(gca, 'XTick', 1:size(MA_list,2)-1)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:end))
set(gca, 'Fontsize', 12)
xtickangle(90)
xlim([0 9])
ylim([0 110])
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
% labels1 = string(1,8);
for i = 1:length(b(1).YData)
    labels1(i) = sprintf('%.0f%%',b(1).YData(i));
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[334   490   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
b.FaceColor = 'flat';
b.CData(8,:) = [200, 14, 80]./255;
box off
% HbR
figure

x_data = MSE_HbR(1,:);
y_data = MSE_HbR(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y,'facecolor',[108, 171, 215]./256);
title('Proportion of trials whose MSE decreased')
set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
ytickformat(gca, 'percentage');
% gca.YGrid = 'on';
set(gca, 'XTick', 1:size(MA_list,2)-1)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:end))
set(gca, 'Fontsize', 12)
xtickangle(90)
xlim([0 9])
ylim([0 110])
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
% labels1 = string(1,8);
for i = 1:length(b(1).YData)
    labels1(i) = sprintf('%.0f%%',b(1).YData(i));
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[665   490   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
b.FaceColor = 'flat';
b.CData(8,:) = [200, 14, 80]./255;
box off
%% calculate of cnr

cnr_HbO = zeros(9,size(HbO,1));
cnr_HbR = zeros(9,size(HbO,1));

cnr_HbO(1,:) = cnr_cal(HbO,HbO_noised,0);
cnr_HbO(2,:) = cnr_cal(HbO,HbO_Spline,0);
cnr_HbO(3,:) = cnr_cal(HbO,HbO_Wavelet05,0);
cnr_HbO(4,:) = cnr_cal(HbO,HbO_Wavelet35,0);
cnr_HbO(5,:) = cnr_cal(HbO,HbO_Kalman,0);
cnr_HbO(6,:) = cnr_cal(HbO,HbO_PCA99,0);
cnr_HbO(7,:) = cnr_cal(HbO,HbO_PCA50,0);
cnr_HbO(8,:) = cnr_cal(HbO,HbO_Cbsi,0);
cnr_HbO(9,:) = cnr_cal(HbO,HbO_NN,0);

cnr_HbR(1,:) = cnr_cal(HbR,HbR_noised,1);
cnr_HbR(2,:) = cnr_cal(HbR,HbR_Spline,1);
cnr_HbR(3,:) = cnr_cal(HbR,HbR_Wavelet05,1);
cnr_HbR(4,:) = cnr_cal(HbR,HbR_Wavelet35,1);
cnr_HbR(5,:) = cnr_cal(HbR,HbR_Kalman,1);
cnr_HbR(6,:) = cnr_cal(HbR,HbR_PCA99,1);
cnr_HbR(7,:) = cnr_cal(HbR,HbR_PCA50,1);
cnr_HbR(8,:) = cnr_cal(HbR,HbR_Cbsi,1);
cnr_HbR(9,:) = cnr_cal(HbR,HbR_NN,1);


%% 3. plot the potion of trials with cnr decreased
figure

x_data = cnr_HbO(1,:);
y_data = cnr_HbO(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y,'facecolor',[108, 171, 215]./256);
title('Proportion of trials whose CNR increased')
set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
ytickformat(gca, 'percentage');
% gca.YGrid = 'on';
set(gca, 'XTick', 1:size(MA_list,2)-1)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:end))
set(gca, 'Fontsize', 12)
xtickangle(90)
xlim([0 9])
ylim([0 110])
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
% labels1 = string(1,8);
for i = 1:length(b(1).YData)
    labels1(i) = sprintf('%.0f%%',b(1).YData(i));
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[996   490   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
b.FaceColor = 'flat';
b.CData(8,:) = [200, 14, 80]./255;
box off
% HbR
figure

x_data = cnr_HbR(1,:);
y_data = cnr_HbR(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y,'facecolor',[108, 171, 215]./256);
title('Proportion of trials whose CNR increased')
set(gca, 'YTick', [0, 50, 100], 'YLim', [0, 100]);
ytickformat(gca, 'percentage');
% gca.YGrid = 'on';
set(gca, 'XTick', 1:size(MA_list,2)-1)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels(2:end))
set(gca, 'Fontsize', 12)
xtickangle(90)
xlim([0 9])
ylim([0 110])
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
% labels1 = string(1,8);
for i = 1:length(b(1).YData)
    labels1(i) = sprintf('%.0f%%',b(1).YData(i));
end
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[3   201   330   215]);
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
b.EdgeColor = [1 1 1];
b.FaceColor = 'flat';
b.CData(8,:) = [200, 14, 80]./255;
box off
%% 4. low ns high noise vol.
% test_idx_HbO = p(n_train+n_val+1:end);
peak_n_list_test = peak_n_list(test_idx,:);
shift_n_list_test = shift_n_list(test_idx,:);
m = size(peak_n_list_test);
peak_n_list_test_HbO = peak_n_list_test(1:m/2);
shift_n_list_test_HbO = shift_n_list_test(1:m/2);

noise_n = peak_n_list_test_HbO + shift_n_list_test_HbO;
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
set(gca, 'XTickLabel', labels(2:8))
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
% test_idx_HbO = p(n_train+n_val+1:end);
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
%% which AUC range is linear with MSE
fs = 7.8125;
MSE_value = MSE_HbO(1,:);
c_c = zeros(10,1);
for i = 1:10
    AUC = abs(trapz(HbO_noised(:,1:round(fs*i*2))./fs,2));
    corrcoef_value = corrcoef(MSE_value,AUC);
    c_c(i) = corrcoef_value(2);
end
figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(2:2:20,c_c,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('Correlation Coefficient')
xlabel('AUC time range (s)')
title('Correlation of MSE and AUC')
set(gca,'FontName','Arial','FontSize',14)
xlim([2 20])
saveas(gcf,'Figures/AUC_examine','fig')
saveas(gcf,'Figures/AUC_examine','svg')
% 18 is the best
%% which AUC ratio range is linear with MSE
fs = 7.8125;
MSE_value = MSE_HbO(1,:);
AUC0_18 = abs(trapz(HbO_noised(:,1:round(fs*18))./fs,2));
c_c = zeros(10,1);
for i = 1:10
    AUC_high = abs(trapz(HbO_noised(:,round(fs*18):round(fs*(18+2*i)))./fs,2));
    AUC_ratio = AUC0_18./AUC_high;
    corrcoef_value = corrcoef(MSE_value,AUC_ratio);
    c_c(i) = corrcoef_value(2);
end
figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(2:2:20,c_c,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('Correlation Coefficient')
xlabel('AUC time range to divide AUC0-18')
title('Correlation of MSE and AUCratio')
set(gca,'FontName','Arial','FontSize',14)
xlim([2 20])
saveas(gcf,'Figures/AUCratio_examine','fig')
saveas(gcf,'Figures/AUCratio_examine','svg')
% 18+8=26 is the best
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
    cnr_mat(i) = (mean(Hb_noised(i,twosecond))-mean(Hb_noised(i,1:round(2*fs))))/std(Hb_noised(i,:)-Hb_data);
end
end