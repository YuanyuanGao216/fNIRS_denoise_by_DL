clear all
close all

%%
global fs_new
fs_new = 7;
global rt
global pt
rt =   40; %resting time 5.7s
pt =   512-40; %performance time 65.536s - 5.7s

%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% load data
load('Processed_data/HRF_test.mat','HRF_test')
load('Processed_data/HRF_test_noised.mat','HRF_test_noised')
time_length = size(HRF_test,2);
t = (1:time_length)/fs_new;

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
STD = 10;
tIncMan=ones(size(t))';
s = zeros(1,time_length);
s((rt):512:time_length) = 1;

filename = 'Processed_data/Sensitivity.txt';
Sensitivity_file = fopen(filename,'w');
%%
sigma_PCA_list = 0.51:0.04:0.99;
mse_list = zeros(length(sigma_PCA_list),m/2);
n_list = zeros(length(sigma_PCA_list),m/2);
for j = 1:length(sigma_PCA_list)
    sigma = sigma_PCA_list(j);
    fprintf('sigma: %f\n',sigma)
    for i = 1:m/2
        dc_HbO = HbO_noised(i,:);dc_HbR = HbR_noised(i,:);
        dc     = [dc_HbO;dc_HbR]';
        [dc_avg,n_MA]       =   proc_PCA(dc, s, SD1, t, tIncMan,STD, sigma);
        dc_predict          =   [dc_avg(:,1)', dc_avg(:,2)'];
        n_list(j,i)         =   n_MA;
        dc_real             =   [HbO(i,1:512),HbR(i,1:512)];
        mse_list(j,i)       =   mean((dc_predict - dc_real).^2);
    end
end
%%
save('Processed_data/sens_PCA.mat','sigma_PCA_list','n_list','mse_list')
plot_sens(sigma_PCA_list,n_list,mse_list,'PCA')

fprintf('PCA:\n')
fprintf(Sensitivity_file,'PCA:\n');

mean_n_list = mean(n_list,2);
fprintf('min n is %f \t sigma is %f\n',min(mean_n_list), sigma_PCA_list(mean_n_list == min(mean_n_list)))
fprintf(Sensitivity_file,'min n is %f \t sigma is %f\n',min(mean_n_list), sigma_PCA_list(mean_n_list == min(mean_n_list)));

mean_mse_list = mean(mse_list,2)*1e9;
fprintf('min mse is %f \t sigma is %f\n',min(mean_mse_list),sigma_PCA_list(mean_mse_list == min(mean_mse_list)))
fprintf(Sensitivity_file,'min mse is %f \t sigma is %f\n',min(mean_mse_list),sigma_PCA_list(mean_mse_list == min(mean_mse_list)));

%%
p_Spline_list = 0:0.01:1;
mse_list = zeros(length(p_Spline_list),m/2);
n_list = zeros(length(p_Spline_list),m/2);

for j = 1:length(p_Spline_list)
    p = p_Spline_list(j);
    fprintf('p: %f\n',p)
    for i = 1:m/2
        dc_HbO = HbO_noised(i,:); dc_HbR  = HbR_noised(i,:);
        dc     = [dc_HbO;dc_HbR]';
        [dc_avg,n_MA]    =   proc_Spline(dc, s, SD1, t, tIncMan, STD, p);
        dc_predict       =   [dc_avg(:,1)', dc_avg(:,2)'];
        n_list(j,i)      =   n_MA;
        dc_real          =   [HbO(i,1:512),HbR(i,1:512)];
        mse_list(j,i)    =   mean((dc_predict - dc_real).^2);
    end
end
%%
save('Processed_data/sens_Spline.mat','p_Spline_list','n_list','mse_list')
plot_sens(p_Spline_list,n_list,mse_list,'Spline')

fprintf('Spline:\n')
fprintf(Sensitivity_file,'Spline:\n');

mean_n_list = mean(n_list,2);
fprintf('min n is %f \t sigma is %f\n',min(mean_n_list), p_Spline_list(mean_n_list == min(mean_n_list)))
fprintf(Sensitivity_file,'min n is %f \t sigma is %f\n',min(mean_n_list), p_Spline_list(mean_n_list == min(mean_n_list)));

mean_mse_list = mean(mse_list,2)*1e9;
fprintf('min mse is %f \t sigma is %f\n',min(mean_mse_list),p_Spline_list(mean_mse_list == min(mean_mse_list)))
fprintf(Sensitivity_file,'min n is %f \t sigma is %f\n',min(mean_n_list), p_Spline_list(mean_n_list == min(mean_n_list)));

%%
alpha_Wavelet = 0.1:0.1:0.4;
mse_list = zeros(length(alpha_Wavelet),m/2);
n_list = zeros(length(alpha_Wavelet),m/2);
for j = 1: length(alpha_Wavelet)
    alpha = alpha_Wavelet(j);
    fprintf('alpha: %f\n',alpha)
    for i = 1:m/2
        dc_HbO  =   HbO_noised(i,:); dc_HbR  =   HbR_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        [dc_avg,n_MA]       =   proc_Wavelet(dc,s,SD1,t,tIncMan,STD,alpha);
        dc_predict          =   [dc_avg(:,1)', dc_avg(:,2)'];
        n_list(j,i)         =   n_MA_Spline;
        dc_real             =   [HbO(i,1:512),HbR(i,1:512)];
        mse_list(j,i)       =   mean((dc_predict - dc_real).^2);
    end
end
%%
save('Processed_data/sens_Wavelet.mat','alpha_Wavelet','n_list','mse_list')
plot_sens(alpha_Wavelet,n_list,mse_list,'Wavelet')

fprintf('Wavelet:\n')
fprintf(Sensitivity_file,'Wavelet:\n');

mean_n_list = mean(n_list,2);
fprintf('min n is %f \t sigma is %f\n',min(mean_n_list), alpha_Wavelet(find(mean_n_list == min(mean_n_list),1)))
fprintf(Sensitivity_file,'min n is %f \t sigma is %f\n',min(mean_n_list), alpha_Wavelet(mean_n_list == min(mean_n_list)));

mean_mse_list = mean(mse_list,2)*1e9;
fprintf('min mse is %f \t sigma is %f\n',min(mean_mse_list),alpha_Wavelet(mean_mse_list == min(mean_mse_list)))
fprintf(Sensitivity_file,'min mse is %f \t sigma is %f\n',min(mean_mse_list),alpha_Wavelet(mean_mse_list == min(mean_mse_list)));

fclose(Sensitivity_file);