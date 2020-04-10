clear all
close all
%% add homer path
% pathHomer = '../../Tools/homer2_src_v2_3_10202017';
% oldpath = cd(pathHomer);
% setpaths;
% cd(oldpath);
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
load('Processed_data/Testing_Wavelet40.mat')
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
HbO_Wavelet05 = HbO_Wavelet05 * 1e6;
HbO_Kalman = HbO_Kalman * 1e6;
HbO_PCA99 = HbO_PCA99 * 1e6;
HbO_PCA50 = HbO_PCA50 * 1e6;
HbO_Cbsi = HbO_Cbsi * 1e6;

HbR_Spline = HbR_Spline * 1e6;
HbR_Wavelet05 = HbR_Wavelet05 * 1e6;
HbR_Wavelet05 = HbR_Wavelet05 * 1e6;
HbR_Kalman = HbR_Kalman * 1e6;
HbR_PCA99 = HbR_PCA99 * 1e6;
HbR_PCA50 = HbR_PCA50 * 1e6;
HbR_Cbsi = HbR_Cbsi * 1e6;

labels = {'No correction','Spline','Wavelet05','Wavelet40','Kalman','PCA99','PCA50','Cbsi','DAE'};

%% n
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

MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet40_HbO,n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet40_HbR,n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];
% 
% 
% figure
% b = bar(MA_list(1,:),'facecolor',[70 116 193]./256);
% ylabel('No. of Motion Artifacts')
% title('Testing dataset')
% set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(45)
% xlim([0 10])
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% set(gcf,'Position',[218   460   330   215]);

%% calculate of mse
MSE_HbO = zeros(9,size(HbO,1));
MSE_HbO(1,:) = mean((HbO-HbO_noised).^2,2);
MSE_HbO(2,:) = mean((HbO-HbO_Spline).^2,2);       % Spline
MSE_HbO(3,:) = mean((HbO-HbO_Wavelet05).^2,2);    % Wavelet05
MSE_HbO(4,:) = mean((HbO-HbO_Wavelet40).^2,2);    % Wavelet40
MSE_HbO(5,:) = mean((HbO-HbO_Kalman).^2,2);       % Kalman
MSE_HbO(6,:) = mean((HbO-HbO_PCA99).^2,2);        % PCA99
MSE_HbO(7,:) = mean((HbO-HbO_PCA50).^2,2);        % PCA50
MSE_HbO(8,:) = mean((HbO-HbO_Cbsi).^2,2);         % Cbsi
MSE_HbO(9,:) = mean((HbO-HbO_NN).^2,2);           % NN

MSE_HbR = zeros(9,size(HbR,1));
MSE_HbR(1,:) = mean((HbR-HbR_noised).^2,2);
MSE_HbR(2,:) = mean((HbR-HbR_Spline).^2,2);       % Spline
MSE_HbR(3,:) = mean((HbR-HbR_Wavelet05).^2,2);    % Wavelet05
MSE_HbR(4,:) = mean((HbR-HbR_Wavelet40).^2,2);    % Wavelet40
MSE_HbR(5,:) = mean((HbR-HbR_Kalman).^2,2);       % Kalman
MSE_HbR(6,:) = mean((HbR-HbR_PCA99).^2,2);        % PCA99
MSE_HbR(7,:) = mean((HbR-HbR_PCA50).^2,2);        % PCA50
MSE_HbR(8,:) = mean((HbR-HbR_Cbsi).^2,2);         % Cbsi
MSE_HbR(9,:) = mean((HbR-HbR_NN).^2,2);           % NN
%% 3. plot the potion of trails with mse decreased
figure
subplot(1,2,1)
x_data = MSE_HbO(1,:);
y_data = MSE_HbO(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y);
title('Proportion of trails whose MSE decreased')
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
subplot(1,2,2)
x_data = MSE_HbR(1,:);
y_data = MSE_HbR(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y);
title('Proportion of trails whose MSE decreased')
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
set(gcf,'Position',[218   460   660   215]);
%% calculate of cnr
cnr_HbO = zeros(9,size(HbO,1));
cnr_HbR = zeros(9,size(HbO,1));
for i = 1:size(HbO,1)
    cnr_HbO(1,i) = cnr_cal(HbO,HbO_noised,0);
    cnr_HbO(2,i) = cnr_cal(HbO,HbO_Spline,0);
    cnr_HbO(3,i) = cnr_cal(HbO,HbO_Wavelet05,0);
    cnr_HbO(4,i) = cnr_cal(HbO,HbO_Wavelet40,0);
    cnr_HbO(5,i) = cnr_cal(HbO,HbO_Kalman,0);
    cnr_HbO(6,i) = cnr_cal(HbO,HbO_PCA99,0);
    cnr_HbO(7,i) = cnr_cal(HbO,HbO_PCA50,0);
    cnr_HbO(8,i) = cnr_cal(HbO,HbO_Cbsi,0);
    cnr_HbO(9,i) = cnr_cal(HbO,HbO_NN,0);
    
    cnr_HbR(1,i) = cnr_cal(HbR,HbR_noised,1);
    cnr_HbR(2,i) = cnr_cal(HbR,HbR_Spline,0);
    cnr_HbR(3,i) = cnr_cal(HbR,HbR_Wavelet05,0);
    cnr_HbR(4,i) = cnr_cal(HbR,HbR_Wavelet40,0);
    cnr_HbR(5,i) = cnr_cal(HbR,HbR_Kalman,0);
    cnr_HbR(6,i) = cnr_cal(HbR,HbR_PCA99,0);
    cnr_HbR(7,i) = cnr_cal(HbR,HbR_PCA50,0);
    cnr_HbR(8,i) = cnr_cal(HbR,HbR_Cbsi,0);
    cnr_HbR(9,i) = cnr_cal(HbR,HbR_NN,0);
end

%% 3. plot the potion of trails with cnr INCREASED
figure
subplot(1,2,1)
x_data = MSE_HbO(1,:);
y_data = MSE_HbO(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y);
title('Proportion of trails whose MSE decreased')
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
subplot(1,2,2)
x_data = MSE_HbR(1,:);
y_data = MSE_HbR(2:end,:);
% n1 = sum(y_data>repmat(x_data,size(y_data,1),1),2);
n2 = sum(y_data<repmat(x_data,size(y_data,1),1),2);
n = size(x_data,2);

y = n2./n*100;
b = bar(y);
title('Proportion of trails whose MSE decreased')
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
set(gcf,'Position',[218   460   660   215]);
%% low ns high noise vol.
peak_n_list_test = peak_n_list(test_idx,:);
shift_n_list_test = shift_n_list(test_idx,:);
noise_n = peak_n_list_test + shift_n_list_test;
med_noise_n = median(noise_n);
low_noise_idx = find(noise_n<=med_noise_n);
high_noise_idx = find(noise_n>med_noise_n);
low_noise_mse = MSE_HbO(2:end,low_noise_idx);
high_noise_mse = MSE_HbO(2:end,high_noise_idx);
val = [mean(low_noise_mse,2),mean(high_noise_mse,2)]';
err  = [std(low_noise_mse,[],2),std(high_noise_mse,[],2)]'./2;
bar([1 2],val)                
hold on
er = errorbar([1 2],val,err);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off

% figure
% subplot(1,2,1)
% boxplot(MSE_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('MSE HbO')
% set(gca, 'XTick', 1:size(MSE_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylabel('\muMol^2')
% 
% subplot(1,2,2)
% boxplot(MSE_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('MSE HbO')
% set(gca, 'XTick', 1:size(MSE_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylabel('\muMol^2')
% set(gcf,'position',[360   511   560   187])
% 
% figure
% subplot(1,2,1)
% boxplot(MSE_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('MSE HbO')
% set(gca, 'XTick', 1:size(MSE_HbO,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylim([0 200])
% ylabel('\muMol^2')
% 
% subplot(1,2,2)
% boxplot(MSE_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('MSE HbO')
% set(gca, 'XTick', 1:size(MSE_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% ylabel('\muMol^2')
% ylim([0 100])
% set(gcf,'position',[360   511   560   187])
% %% finally 1. plot the the boxplot of R2
% R2_HbO = zeros(9,size(HbO,1));
% R2_HbO(1,:) = 1 - sum((HbO-HbO_noised).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(2,:) = 1 - sum((HbO-HbO_Spline).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(3,:) = 1 - sum((HbO-HbO_Wavelet05).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(4,:) = 1 - sum((HbO-HbO_Wavelet40).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(5,:) = 1 - sum((HbO-HbO_Kalman).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(6,:) = 1 - sum((HbO-HbO_PCA99).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(7,:) = 1 - sum((HbO-HbO_PCA50).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(8,:) = 1 - sum((HbO-HbO_Cbsi).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% R2_HbO(9,:) = 1 - sum((HbO-HbO_NN).^2,2)./sum((HbO-repmat(mean(HbO,2),1,512)).^2,2);
% 
% R2_HbR = zeros(9,size(HbR,1));
% R2_HbR(1,:) = 1 - sum((HbR-HbR_noised).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(2,:) = 1 - sum((HbR-HbR_Spline).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(3,:) = 1 - sum((HbR-HbR_Wavelet05).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(4,:) = 1 - sum((HbR-HbR_Wavelet40).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(5,:) = 1 - sum((HbR-HbR_Kalman).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(6,:) = 1 - sum((HbR-HbR_PCA99).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(7,:) = 1 - sum((HbR-HbR_PCA50).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(8,:) = 1 - sum((HbR-HbR_Cbsi).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% R2_HbR(9,:) = 1 - sum((HbR-HbR_NN).^2,2)./sum((HbR-repmat(mean(HbR,2),1,512)).^2,2);
% 
% figure
% subplot(1,2,1)
% boxplot(R2_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('R^2 HbO')
% set(gca, 'XTick', 1:size(R2_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% 
% 
% subplot(1,2,2)
% boxplot(R2_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('R^2 HbO')
% set(gca, 'XTick', 1:size(R2_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% set(gcf,'position',[360   511   560   187])
% 
% %% finally 1. plot the the boxplot of CNR
% R2_HbO = zeros(9,size(HbO,1));
% R2_HbO(1,:) = 1 - sum((HbO-HbO_noised).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(2,:) = 1 - sum((HbO-HbO_Spline).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(3,:) = 1 - sum((HbO-HbO_Wavelet05).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(4,:) = 1 - sum((HbO-HbO_Wavelet40).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(5,:) = 1 - sum((HbO-HbO_Kalman).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(6,:) = 1 - sum((HbO-HbO_PCA99).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(7,:) = 1 - sum((HbO-HbO_PCA50).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(8,:) = 1 - sum((HbO-HbO_Cbsi).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% R2_HbO(9,:) = 1 - sum((HbO-HbO_NN).^2,'all')./sum((HbO-repmat(mean(HbO,2),1,512)).^2,'all');
% 
% R2_HbR = zeros(9,size(HbR,1));
% R2_HbR(1,:) = 1 - sum((HbR-HbR_noised).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(2,:) = 1 - sum((HbR-HbR_Spline).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(3,:) = 1 - sum((HbR-HbR_Wavelet05).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(4,:) = 1 - sum((HbR-HbR_Wavelet40).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(5,:) = 1 - sum((HbR-HbR_Kalman).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(6,:) = 1 - sum((HbR-HbR_PCA99).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(7,:) = 1 - sum((HbR-HbR_PCA50).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(8,:) = 1 - sum((HbR-HbR_Cbsi).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% R2_HbR(9,:) = 1 - sum((HbR-HbR_NN).^2,'all')./sum((HbR-repmat(mean(HbR,2),1,512)).^2,'all');
% 
% figure
% subplot(1,2,1)
% boxplot(R2_HbO','colors','k','OutlierSize',2,'Symbol','b.')
% title('R^2 HbO')
% set(gca, 'XTick', 1:size(R2_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% 
% 
% subplot(1,2,2)
% boxplot(R2_HbR','colors','k','OutlierSize',2,'Symbol','b.')
% title('R^2 HbO')
% set(gca, 'XTick', 1:size(R2_HbR,1),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% set(gcf,'position',[360   511   560   187])
%% 2. plot the dot plot of mse, R2, CNR and N for all methods
% fontsize = 12;
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = MSE_HbO(1,:);
%     y_data = MSE_HbO(i+1,:);
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
% figure
% for i = 1:8
%     subplot(4,2,i)
%     x_data = MSE_HbR(1,:);
%     y_data = MSE_HbR(i+1,:);
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


function cnr = cnr_cal(Hb,Hb_noised,flag)
if flag == 0
    max_index_Hb = find(HbO == max(HbO));
else
    max_index_Hb = find(HbO == min(HbO));
end
twosecond = max_index_Hb-round(1*fs):max_index_Hb+round(1*fs);
cnr = (mean(Hb_noised(twosecond))-mean(Hb_noised(1:round(2*fs))))/std(Hb_noised'-Hb);
end
