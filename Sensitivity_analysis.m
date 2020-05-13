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
%% 
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
%%
sigma_PCA_list = 0.51:0.04:0.99;
MSE_PCA = 1e6;
R2_PCA = 0;
CNR_PCA = 0;
n_left_PCA = 1e6;
MSE_PCA_j = 0;
R2_PCA_j = 0;
CNR_PCA_j = 0;
n_left_PCA_j = 0;
r2_list = [];
mse_list = [];
cnr_list = [];
n_list = [];
for sigma = sigma_PCA_list
% for sigma = [0.51 0.99]
    fprintf('sigma: %f\n',sigma)
    mse = 0;
    sse = 0;
    sst = 0;
    cnr = 0;
    n = 0;
    for i = 1:m/2
%     for i = 100:100
        dc_HbO  =   HbO_noised(i,:);
        dc_HbR  =   HbR_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        dc_real = [HbO(i,:),HbR(i,:)];
        
%         figure
%         hold on
%         plot([dc_HbO,dc_HbR],'b')
%         plot(dc_real,'k')
%         
        dod     =   hmrConc2OD( dc, SD1, ppf );
        [dod_PCA,~,~,~,~]   =   hmrMotionCorrectPCArecurse(dod,t,SD1,tIncMan,0.5,1,STD,200,sigma,5);
        [~,tIncAuto_PCA]    =   hmrMotionArtifactByChannel(dod_PCA,t,SD1,tIncMan,0.5,1,STD,200);
        dod_PCA             =   hmrBandpassFilt(dod_PCA,t,0,0.5);
        dc_PCA              =   hmrOD2Conc(dod_PCA,SD1,[6  6]);
        dc_predict_HbO = squeeze(dc_PCA(:,1,:));
        dc_predict_HbR = squeeze(dc_PCA(:,2,:));
        dc_predict = [dc_predict_HbO',dc_predict_HbR'];
        
%         plot(dc_predict,'r')
        n_MA_PCA = 0;
        for Ch = 1:2
            [n_MA_PCA_Ch,~,~]    =   CalMotionArtifact(tIncAuto_PCA(:,Ch));
            n_MA_PCA = n_MA_PCA + n_MA_PCA_Ch;
        end
        n   = n + n_MA_PCA;
        mse = mse + sum((dc_predict-dc_real).^2);
        sse = sse + sum((dc_predict-dc_real).^2);
        sst = sst + sum((mean(dc_real)-dc_real).^2);
        
        max_index_HbO = find(HbO(i,:) == max(HbO(i,:)));
        twosecond_HbO = max_index_HbO-round(1*fs):max_index_HbO+round(1*fs);
        min_index_HbR = find(HbR(i,:) == min(HbR(i,:)));
        twosecond_HbR = min_index_HbR-round(1*fs):min_index_HbR+round(1*fs);
        cnr_HbO = abs(mean(dc_predict_HbO(twosecond_HbO))-mean(dc_predict_HbO(1:round(2*fs))))/std(dc_predict_HbO'-HbO(i,:));
        cnr_HbR = abs(mean(dc_predict_HbR(twosecond_HbR))-mean(dc_predict_HbR(1:round(2*fs))))/std(dc_predict_HbR'-HbR(i,:));
        cnr = cnr + cnr_HbO + cnr_HbR;
        
%         fprintf('mse is %f\t sse is %f\t sst is %f\t r2 is %f\t cnr is %f\n',mse*1e6,sse*1e6,sst*1e6,1-sse/sst,cnr)
    end
    r2 = 1-sse/sst;
    fprintf('mse is %f\t r2 is %f\t cnr is %f\t n is %d\n',mse*1e6,1-sse/sst,cnr,n)
    r2_list(end+1) = r2;
    mse_list(end+1) = mse;
    cnr_list(end+1) = cnr;
    n_list(end+1) = n;
    if mse < MSE_PCA
        MSE_PCA = mse;
        MSE_PCA_j = sigma;
    end
    if r2 > R2_PCA
        R2_PCA = r2;
        R2_PCA_j = sigma;
    end
    if cnr > CNR_PCA
        CNR_PCA = cnr;
        CNR_PCA_j = sigma;
    end
    if n < n_left_PCA
        n_left_PCA = n;
        n_left_PCA_j = sigma;
    end
end
%%
figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(sigma_PCA_list,r2_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('R2')
xlabel('sigma')
title('PCA')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/PCA_r2','fig')
saveas(gcf,'Figures/PCA_r2','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(sigma_PCA_list,mse_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('mse')
xlabel('sigma')
title('PCA')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/PCA_mse','fig')
saveas(gcf,'Figures/PCA_mse','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(sigma_PCA_list,cnr_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('cnr')
xlabel('sigma')
title('PCA')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/PCA_cnr','fig')
saveas(gcf,'Figures/PCA_cnr','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(sigma_PCA_list,n_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('n')
xlabel('sigma')
title('PCA')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/PCA_n','fig')
saveas(gcf,'Figures/PCA_n','svg')

%%
p_Spline_list = 0:0.01:1;
MSE_Spline = 1e6;
R2_Spline = 0;
CNR_Spline = 0;
n_left_Spline = 1e6;
MSE_Spline_j = 0;
R2_Spline_j = 0;
CNR_Spline_j = 0;
n_left_Spline_j = 0;
r2_list = [];
mse_list = [];
cnr_list = [];
n_list = [];
for p = p_Spline_list
    fprintf('p: %f\n',p)
    mse = 0;
    sse = 0;
    sst = 0;
    cnr = 0;
    n = 0;
    for i = 1:m/2
        dc_HbO  =   HbO_noised(i,:);
        dc_HbR  =   HbR_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        dc_real = [HbO(i,:),HbR(i,:)];
        dod     =   hmrConc2OD( dc, SD1, ppf );
        [~,tIncChAuto]    =   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,200);
        [dod_Spline]      =   hmrMotionCorrectSpline(dod,t,SD1,tIncChAuto,p);
        [~,tIncAuto_Spline]    =   hmrMotionArtifactByChannel(dod_Spline,t,SD1,tIncMan,0.5,1,STD,200);
        dod_Spline        =   hmrBandpassFilt(dod_Spline,t,0,0.5);
        dc_Spline         =   hmrOD2Conc(dod_Spline,SD1,[6  6]);
        dc_predict_HbO = squeeze(dc_Spline(:,1,:));
        dc_predict_HbR = squeeze(dc_Spline(:,2,:));
        dc_predict = [dc_predict_HbO',dc_predict_HbR'];
        
        n_MA_Spline = 0;
        for Ch = 1:2
            [n_MA_Spline_Ch,~,~]    =   CalMotionArtifact(tIncAuto_Spline(:,Ch));
            n_MA_Spline = n_MA_Spline + n_MA_Spline_Ch;
        end
        n   = n + n_MA_Spline;
        mse = mse + sum((dc_predict-dc_real).^2);
        sse = sse + sum((dc_predict-dc_real).^2);
        sst = sst + sum((mean(dc_real)-dc_real).^2);
        
        max_index_HbO = find(HbO(i,:) == max(HbO(i,:)));
        twosecond_HbO = max_index_HbO-round(1*fs):max_index_HbO+round(1*fs);
        min_index_HbR = find(HbR(i,:) == min(HbR(i,:)));
        twosecond_HbR = min_index_HbR-round(1*fs):min_index_HbR+round(1*fs);
        cnr_HbO = abs(mean(dc_predict_HbO(twosecond_HbO))-mean(dc_predict_HbO(1:round(2*fs))))/std(dc_predict_HbO'-HbO(i,:));
        cnr_HbR = abs(mean(dc_predict_HbR(twosecond_HbR))-mean(dc_predict_HbR(1:round(2*fs))))/std(dc_predict_HbR'-HbR(i,:));
        cnr = cnr + cnr_HbO + cnr_HbR;
        
    end
    r2 = 1-sse/sst;
    fprintf('mse is %f\t r2 is %f\t cnr is %f\t n is %d\n',mse*1e6,1-sse/sst,cnr,n)
    r2_list(end+1) = r2;
    mse_list(end+1) = mse;
    cnr_list(end+1) = cnr;
    n_list(end+1) = n;
    if mse < MSE_Spline
        MSE_Spline = mse;
        MSE_Spline_j = p;
    end
    if r2 > R2_Spline
        R2_Spline = r2;
        R2_Spline_j = p;
    end
    if cnr > CNR_Spline
        CNR_Spline = cnr;
        CNR_Spline_j = p;
    end
    if n < n_left_Spline
        n_left_Spline = n;
        n_left_Spline_j = p;
    end
end
figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(p_Spline_list,r2_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('R2')
xlabel('p')
title('Spline')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Spline_r2','fig')
saveas(gcf,'Figures/Spline_r2','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(p_Spline_list,mse_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('mse')
xlabel('p')
title('Spline')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Spline_mse','fig')
saveas(gcf,'Figures/Spline_mse','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(p_Spline_list,cnr_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('cnr')
xlabel('p')
title('Spline')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Spline_cnr','fig')
saveas(gcf,'Figures/Spline_cnr','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(p_Spline_list,n_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('n')
xlabel('p')
title('Spline')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Spline_n','fig')
saveas(gcf,'Figures/Spline_n','svg')
%%
alpha_Wavelet = 0.05:0.05:0.4;
MSE_Wavelet = 1e6;
R2_Wavelet = 0;
CNR_Wavelet = 0;
n_left_Wavelet = 1e6;
MSE_Wavelet_j = 0;
R2_Wavelet_j = 0;
CNR_Wavelet_j = 0;
n_left_Wavelet_j = 0;
r2_list = [];
mse_list = [];
cnr_list = [];
n_list = [];
for alpha = alpha_Wavelet
    fprintf('alpha: %f\n',alpha)
    mse = 0;
    sse = 0;
    sst = 0;
    cnr = 0;
    n = 0;
    for i = 1:m/2
        
        dc_HbO  =   HbO_noised(i,:);
        dc_HbR  =   HbR_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        dc_real =   [HbO(i,:),HbR(i,:)];
        dod     =   hmrConc2OD( dc, SD1, ppf );
        [dod_Wavelet]           =   hmrMotionCorrectWavelet(dod,SD1,alpha);
        [~,tIncAuto_Wavelet]    =   hmrMotionArtifactByChannel(dod_Wavelet,t,SD1,tIncMan,0.5,1,STD,200);
        dod_Wavelet             =   hmrBandpassFilt(dod_Wavelet,t,0,0.5);
        dc_Wavelet              =   hmrOD2Conc(dod_Wavelet,SD1,[6  6]);
        dc_predict_HbO = squeeze(dc_Wavelet(:,1,:));
        dc_predict_HbR = squeeze(dc_Wavelet(:,2,:));
        dc_predict = [dc_predict_HbO',dc_predict_HbR'];
        
        
        n_MA_Wavelet = 0;
        for Ch = 1:2
            [n_MA_Wavelet_Ch,~,~]    =   CalMotionArtifact(tIncAuto_Wavelet(:,Ch));
            n_MA_Wavelet = n_MA_Wavelet + n_MA_Wavelet_Ch;
        end
        n   = n + n_MA_Wavelet;
        mse = mse + sum((dc_predict-dc_real).^2);
        sse = sse + sum((dc_predict-dc_real).^2);
        sst = sst + sum((mean(dc_real)-dc_real).^2);
        
        max_index_HbO = find(HbO(i,:) == max(HbO(i,:)));
        twosecond_HbO = max_index_HbO-round(1*fs):max_index_HbO+round(1*fs);
        min_index_HbR = find(HbR(i,:) == min(HbR(i,:)));
        twosecond_HbR = min_index_HbR-round(1*fs):min_index_HbR+round(1*fs);
        cnr_HbO = abs(mean(dc_predict_HbO(twosecond_HbO))-mean(dc_predict_HbO(1:round(2*fs))))/std(dc_predict_HbO'-HbO(i,:));
        cnr_HbR = abs(mean(dc_predict_HbR(twosecond_HbR))-mean(dc_predict_HbR(1:round(2*fs))))/std(dc_predict_HbR'-HbR(i,:));
        cnr = cnr + cnr_HbO + cnr_HbR;
        
    end
    r2 = 1-sse/sst;
    fprintf('mse is %f\t r2 is %f\t cnr is %f\t n is %d\n',mse*1e6,1-sse/sst,cnr,n)
    r2_list(end+1) = r2;
    mse_list(end+1) = mse;
    cnr_list(end+1) = cnr;
    n_list(end+1) = n;
    if mse < MSE_Wavelet
        MSE_Wavelet = mse;
        MSE_Wavelet_j = alpha;
    end
    if r2 > R2_Wavelet
        R2_Wavelet = r2;
        R2_Wavelet_j = alpha;
    end
    if cnr > CNR_Wavelet
        CNR_Wavelet = cnr;
        CNR_Wavelet_j = alpha;
    end
    if n < n_left_Wavelet
        n_left_Wavelet = n;
        n_left_Wavelet_j = alpha;
    end
end
figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(alpha_Wavelet,r2_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('R2')
xlabel('alpha')
title('Wavelet')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Wavelet_r2','fig')
saveas(gcf,'Figures/Wavelet_r2','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(alpha_Wavelet,mse_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('mse')
xlabel('alpha')
title('Wavelet')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Wavelet_mse','fig')
saveas(gcf,'Figures/Wavelet_mse','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(alpha_Wavelet,cnr_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('cnr')
xlabel('alpha')
title('Wavelet')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Wavelet_cnr','fig')
saveas(gcf,'Figures/Wavelet_cnr','svg')

figure('Renderer', 'painters', 'Position', [10 10 300 200])
plot(alpha_Wavelet,n_list,'Linewidth',1,'Marker','o','markerfacecolor','b')
ylabel('n')
xlabel('alpha')
title('Wavelet')
set(gca,'FontName','Times New Roman','FontSize',15)
saveas(gcf,'Figures/Wavelet_n','fig')
saveas(gcf,'Figures/Wavelet_n','svg')



fprintf('PCA:\n')
fprintf('mse is %f \t sigma is %f\n',MSE_PCA*1e6, MSE_PCA_j)
fprintf('r2 is %f \t sigma is %f\n',R2_PCA,R2_PCA_j)
fprintf('cnr is %f \t sigma is %f\n',CNR_PCA,CNR_PCA_j)
fprintf('n is %f \t sigma is %f\n',n_left_PCA,n_left_PCA_j)
fprintf('Spline:\n')
fprintf('mse is %f \t sigma is %f\n',MSE_Spline*1e6, MSE_Spline_j)
fprintf('r2 is %f \t sigma is %f\n',R2_Spline,R2_Spline_j)
fprintf('cnr is %f \t sigma is %f\n',CNR_Spline,CNR_Spline_j)
fprintf('n is %f \t sigma is %f\n',n_left_Spline,n_left_Spline_j)
fprintf('Wavelet:\n')
fprintf('mse is %f \t sigma is %f\n',MSE_Wavelet*1e6, MSE_Wavelet_j)
fprintf('r2 is %f \t sigma is %f\n',R2_Wavelet,R2_Wavelet_j)
fprintf('cnr is %f \t sigma is %f\n',CNR_Wavelet,CNR_Wavelet_j)
fprintf('n is %f \t sigma is %f\n',n_left_Wavelet,n_left_Wavelet_j)

filename = 'Processed_data/Sensitivity.txt';
Sensitivity_file = fopen(filename,'w');

fprintf(Sensitivity_file,'PCA:\n');
fprintf(Sensitivity_file,'mse is %f \t sigma is %f\n',MSE_PCA*1e6, MSE_PCA_j);
fprintf(Sensitivity_file,'r2 is %f \t sigma is %f\n',R2_PCA,R2_PCA_j);
fprintf(Sensitivity_file,'cnr is %f \t sigma is %f\n',CNR_PCA,CNR_PCA_j);
fprintf(Sensitivity_file,'n is %f \t sigma is %f\n',n_left_PCA,n_left_PCA_j);
fprintf(Sensitivity_file,'Spline:\n');
fprintf(Sensitivity_file,'mse is %f \t sigma is %f\n',MSE_Spline*1e6, MSE_Spline_j);
fprintf(Sensitivity_file,'r2 is %f \t sigma is %f\n',R2_Spline,R2_Spline_j);
fprintf(Sensitivity_file,'cnr is %f \t sigma is %f\n',CNR_Spline,CNR_Spline_j);
fprintf(Sensitivity_file,'n is %f \t sigma is %f\n',n_left_Spline,n_left_Spline_j);
fprintf(Sensitivity_file,'Wavelet:\n');
fprintf(Sensitivity_file,'mse is %f \t sigma is %f\n',MSE_Wavelet*1e6, MSE_Wavelet_j);
fprintf(Sensitivity_file,'r2 is %f \t sigma is %f\n',R2_Wavelet,R2_Wavelet_j);
fprintf(Sensitivity_file,'cnr is %f \t sigma is %f\n',CNR_Wavelet,CNR_Wavelet_j);
fprintf(Sensitivity_file,'n is %f \t sigma is %f\n',n_left_Wavelet,n_left_Wavelet_j);

fclose(Sensitivity_file);