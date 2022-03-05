clear all

%% SNR of real data and the simulated data
% load real data
load('Processed_data/RealData.mat','net_input')
% snr_real_mat = zeros(length(net_input)*14,2);
snr_real_mat = zeros(1*14,2);
k = 1;
for i = 1:length(net_input)
% for i = 1:1
    dc = net_input(i).dc;
    for j = 1:14
        HbO = squeeze(dc(:,1,j));
        HbR = squeeze(dc(:,2,j));
        snr_real_mat(k,1) = snr(HbO);
        snr_real_mat(k,2) = snr(HbR);
        k = k + 1;
    end
end

mean(snr_real_mat,1)
median(snr_real_mat,1)
figure;histogram(snr_real_mat)
xlabel('SNR')
ylabel('N')
title('Experimental data')
set(gcf,'position',[10         10         252         190])
set(gca, 'FontName', 'Arial','fontsize',10)
xlim([-20 20])
% load sim data
%noise
load('Processed_data/leave_1_out/Noise.mat','Noise_train','Noise_val','Noise_test')
% Resting
load('Processed_data/leave_1_out/Resting.mat','Resting_train','Resting_val','Resting_test')
% sim data with only resting and noise
Train_data = Noise_train + Resting_train;
Val_data = Noise_val + Resting_val;
Test_data = Noise_test + Resting_test;

simulated_HbO = [Train_data(1:size(Train_data,1)/2,:); Val_data(1:size(Val_data,1)/2,:); Test_data(1:size(Test_data,1)/2,:)];
simulated_HbR = [Train_data(size(Train_data,1)/2+1:end,:); Val_data(size(Val_data,1)/2+1:end,:); Test_data(size(Test_data,1)/2+1:end,:)];

snr_sim_mat = zeros(size(simulated_HbO,1),2);
for i = 1:size(simulated_HbO,1)
    HbO = simulated_HbO(i,:);
    HbR = simulated_HbR(i,:);
    snr_sim_mat(i,1) = snr(HbO);
    snr_sim_mat(i,2) = snr(HbR);
end

mean(snr_sim_mat,1)
median(snr_sim_mat,1)
figure;histogram(snr_sim_mat,10)
xlabel('SNR')
ylabel('N')
title('Simulated data')
set(gcf,'position',[10         10         252         190])
set(gca, 'FontName', 'Arial','fontsize',10)
xlim([-20 20])


%% adding not HRF data into the simulated data
