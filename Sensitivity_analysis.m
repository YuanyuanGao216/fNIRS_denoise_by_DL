clear all
close all

%% define constants
define_constants

%% add homer path
pathHomer = 'Tools/homer2_src_v2_3_10202017/';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% load data since it is leave one out so we need to load every test dataset
DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
HRF_test_mat = [];
HRF_test_noised_mat = [];
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
    sim_path = fullfile(DataDir, subfolders(subfolder).name, 'SimulateData.mat');
    %% load data
    load(sim_path,'HRF_test','HRF_test_noised')
    HRF_test_mat = [HRF_test_mat; HRF_test];
    HRF_test_noised_mat = [HRF_test_noised_mat; HRF_test_noised];
end
k = randperm(size(HRF_test_mat,1));
HRF_test = HRF_test_mat(k(1:1000),:);
HRF_test_noised = HRF_test_noised_mat(k(1:1000),:);
clear HRF_test_mat HRF_test_noised_mat

tp = size(HRF_test,2);
t = (1:tp)/fs_new;

[m,n] = size(HRF_test_noised);
HbO_noised = HRF_test_noised(1:m/2,:);
HbR_noised = HRF_test_noised(m/2+1:end,:);
HbO = HRF_test(1:m/2,:);
HbR = HRF_test(m/2+1:end,:);
%%  1 pair of source and detector
SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];

tIncMan=ones(size(t))';
s = zeros(1,tp);
s((rt):512:tp) = 1;

fclose('all')
filename = 'Processed_data/Sensitivity.txt';
Sensitivity_file = fopen(filename,'w');
%%
% sigma_PCA_list = 0.51:0.01:0.99;
% mse_list = zeros(length(sigma_PCA_list),m/2);
% n_list = zeros(length(sigma_PCA_list),m/2);
% for j = 1:length(sigma_PCA_list)
%     sigma = sigma_PCA_list(j);
%     fprintf('sigma: %f\n',sigma)
%     for i = 1:m/2
%         dc_HbO              =   HbO_noised(i,:);
%         dc_HbR              =   HbR_noised(i,:);
%         dc                  =   [dc_HbO;dc_HbR]';
%         [dc_avg,n_MA]       =   proc_PCA(dc, s, SD1, t, tIncMan, STD, OD_thred, sigma);
%         dc_predict          =   [dc_avg(:,1)', dc_avg(:,2)'];
%         n_list(j,i)         =   n_MA;
%         dc_real             =   [HbO(i,1:512),HbR(i,1:512)];
%         mse_list(j,i)       =   mean((dc_predict - dc_real).^2);
%     end
% end
% %%
% save('Processed_data/sens_PCA.mat','sigma_PCA_list','n_list','mse_list')
% plot_sens(sigma_PCA_list,n_list,mse_list,'PCA','sigma')
% 
% fprintf('PCA:\n')
% fprintf(Sensitivity_file,'PCA:\n');
% 
% mean_n_list = mean(n_list,2);
% 
% for k = 1:length(mean_n_list)
%     fprintf('n is %f \t sigma is %f\n',mean_n_list(k), sigma_PCA_list(k))
%     fprintf(Sensitivity_file,'n is %f \t sigma is %f\n',mean_n_list(k), sigma_PCA_list(k));
% end
% 
% mean_mse_list = mean(mse_list,2)*1e9;
% for k = 1:length(mean_mse_list)
%     fprintf('mse is %f \t sigma is %f\n',mean_mse_list(k),sigma_PCA_list(k))
%     fprintf(Sensitivity_file,'mse is %f \t sigma is %f\n',mean_mse_list(k),sigma_PCA_list(k));
% end
% %%
p_Spline_list = 0:0.01:1;
mse_list = zeros(length(p_Spline_list),m/2);
n_list = zeros(length(p_Spline_list),m/2);

for j = 1:length(p_Spline_list)
    p = p_Spline_list(j);
    fprintf('p: %f\n',p)
    for i = 1:m/2
        dc_HbO           =   HbO_noised(i,:); dc_HbR  = HbR_noised(i,:);
        dc               =   [dc_HbO;dc_HbR]';
        [dc_avg,n_MA]    =   proc_Spline(dc, s, SD1, t, tIncMan, STD, OD_thred,p);
        dc_predict       =   [dc_avg(:,1)', dc_avg(:,2)'];
        n_list(j,i)      =   n_MA;
        dc_real          =   [HbO(i,1:512),HbR(i,1:512)];
        mse_list(j,i)    =   mean((dc_predict - dc_real).^2);
    end
end
%%
save('Processed_data/sens_Spline.mat','p_Spline_list','n_list','mse_list')
plot_sens(p_Spline_list,n_list,mse_list,'Spline','p')

fprintf('Spline:\n')
fprintf(Sensitivity_file,'Spline:\n');

mean_n_list = mean(n_list,2);
for k = 1:length(mean_n_list)
    fprintf( 'n is %f \t p is %f\n',mean_n_list(k), p_Spline_list(k))
    fprintf(Sensitivity_file,'n is %f \t p is %f\n',mean_n_list(k), p_Spline_list(k));
end

mean_mse_list = mean(mse_list,2)*1e9;
for k = 1:length(mean_mse_list)
    fprintf('mse is %f \t p is %f\n',mean_mse_list(k),p_Spline_list(k))
    fprintf(Sensitivity_file,'mse is %f \t p is %f\n',mean_mse_list(k),p_Spline_list(k));
end


%%
iqr_Wavelet = [0.1:0.05:0.4 0.5:0.25:1.5];
mse_list = zeros(length(iqr_Wavelet),m/2);
n_list = zeros(length(iqr_Wavelet),m/2);
for j = 1: length(iqr_Wavelet)
    iqr = iqr_Wavelet(j);
    fprintf('iqr: %f\n',iqr)
    for i = 1:m/2
        dc_HbO              =   HbO_noised(i,:); dc_HbR  =   HbR_noised(i,:);
        dc                  =   [dc_HbO;dc_HbR]';
        [dc_avg,n_MA]       =   proc_Wavelet(dc,s,SD1,t,tIncMan,STD,OD_thred,iqr);
        dc_predict          =   [dc_avg(:,1)', dc_avg(:,2)'];
        n_list(j,i)         =   n_MA;
        dc_real             =   [HbO(i,1:512),HbR(i,1:512)];
        mse_list(j,i)       =   mean((dc_predict - dc_real).^2);
    end
end
%%
save('Processed_data/sens_Wavelet.mat','iqr_Wavelet','n_list','mse_list')
plot_sens(iqr_Wavelet,n_list,mse_list,'Wavelet','iqr')

fprintf('Wavelet:\n')
fprintf(Sensitivity_file,'Wavelet:\n');

mean_n_list = mean(n_list,2);
for k = 1:length(mean_n_list)
    fprintf( 'n is %f \t iqr is %f\n',mean_n_list(k), iqr_Wavelet(k))
    fprintf(Sensitivity_file,'n is %f \t iqr is %f\n',mean_n_list(k), iqr_Wavelet(k));
end

mean_mse_list = mean(mse_list,2)*1e9;
for k = 1:length(mean_mse_list)
    fprintf('mse is %f \t iqr is %f\n',mean_mse_list(k),iqr_Wavelet(k))
    fprintf(Sensitivity_file,'mse is %f \t iqr is %f\n',mean_mse_list(k),iqr_Wavelet(k));
end


fclose(Sensitivity_file);

% (sigma_PCA is 0.97 in Cooper 2012, 0.97 and 0.80 in Brigadoi et al., 2014) Here I will say 0.97
% for Spline, it is 0.99 in Brigadoi et al., 2014; Cooper et al., 2012; Scholkmann et al., 2010
% For wavelet, iqr = 0.10 selected in Brigadoi et al., 2014; Cooper et al.,2012; Molavi and Dumont, 2012