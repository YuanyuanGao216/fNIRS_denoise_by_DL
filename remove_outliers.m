% remove some outliers which have much higher values, which make the
% network very unstable
% function remove_outliers()
%% load datasets
load('Processed_data/noised_HRF_matrix.mat','noised_HRF_matrix')%input
load('Processed_data/noise_profile.mat','noise_profile')
load('Processed_data/HRF_profile.mat','HRF_profile')%output
load('Real_HbO.mat')
load('Real_HbR.mat')
%% find outliers
HRF_sum = sum(noised_HRF_matrix,2);
Real_HbO_sum = sum(Real_HbO,2);
Real_HbR_sum = sum(Real_HbR,2);
thres = max([max(abs(Real_HbO_sum)),max(abs(Real_HbR_sum))]);
index = find(abs(HRF_sum)>thres);
noised_HRF_matrix(index,:) = [];
noise_profile(index,:) = [];
HRF_profile(index,:) = [];

%% plot
HRF_sum = sum(noised_HRF_matrix,2);
figure
plot(HRF_sum)
%% save
save('Processed_data/noise_profile.mat','noise_profile')
save('Processed_data/HRF_profile.mat','HRF_profile')%output
save('Processed_data/noised_HRF_matrix.mat','noised_HRF_matrix')%input