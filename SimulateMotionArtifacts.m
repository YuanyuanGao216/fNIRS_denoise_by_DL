% function SimulateMotionArtifacts()
clear all
%% load data
load('Processed_data/simulated_HbO.mat','simulated_HbO')
load('Processed_data/simulated_HbR.mat','simulated_HbR')
load('Processed_data/pds.mat')
load('Processed_data/Real_HbO.mat')
load('Processed_data/Real_HbR.mat')
%%
% simu = gamrnd(pd_HbO.a,pd_HbO.b,1500,1);
% figure
% n_dist_HbO = round(gamrnd(pd_n_HbO.a,pd_n_HbO.b,length(simulated_HbO),1));

%%
Resting_matrix = [simulated_HbO;simulated_HbR];
n_dist_HbO = round(gamrnd(pd_n_HbO.a,pd_n_HbO.b,length(simulated_HbO),1));
n_dist_HbR = round(gamrnd(pd_n_HbR.a,pd_n_HbR.b,length(simulated_HbR),1));
n_list = [n_dist_HbO;n_dist_HbR];
noised_HRF_matrix = zeros(size(Resting_matrix));
noise_profile = zeros(size(Resting_matrix));
HRF_profile = zeros(size(Resting_matrix));
fs = 7.8125;
pt = 512/fs;
t = 1:fs*pt;
seed = 101;
rng(seed);
for i = 1:length(Resting_matrix)
    %HRF
    Resting = Resting_matrix(i,:);
    N = n_list(i);
    noise = zeros(size(Resting));
    if N~=0
        for j = 1:N
            r = randi([0 1],1,1);
            neg_sign = randi([-1 1],1,1);
            if r == 0
                t0 = fs*pt*rand;
                if i <= length(simulated_HbO)
                    A = gamrnd(pd_diff_HbO.a,pd_diff_HbO.b,1,1);
                else
                    A = gamrnd(pd_diff_HbR.a,pd_diff_HbR.b,1,1);
                end
                b_low = 0;b_high = 1.5;%b is from A_low to A_high
                b = b_low + rand.*(b_high-b_low);%b is from 0 to 1.5
                noise = noise + A.*exp(-abs(t-t0)./(b*fs));
                noise = neg_sign * noise;
            else
                transition = round(0.25+(1.5-0.25)*rand);%shift transition time is from 0.25s to 1.5s
                start_point = round(1+(fs*pt-transition*fs-1-1)*rand);%from 1 to fs*pt-transition*25-1
                end_point = start_point+transition*fs;
                if i <= length(simulated_HbO)
                    DC_shift = gamrnd(pd_diff_HbO.a,pd_diff_HbO.b,1,1);
                else
                    DC_shift = gamrnd(pd_diff_HbR.a,pd_diff_HbR.b,1,1);
                end
                shift_sim = zeros(fs*pt,1);
                shift_sim(start_point:end_point) = linspace(0,DC_shift,transition*fs+1);
                shift_sim(end_point:end) = DC_shift;
                noise = noise + shift_sim';
                noise = neg_sign * noise;
            end
        end
    end
    if i <= length(simulated_HbO)
        amp_Hb = gamrnd(pd_HRF_HbO.a,pd_HRF_HbO.b,1,1);
        HRF = amp_Hb./gamma(t/15/fs);
    else
        amp_Hb = gamrnd(pd_HRF_HbR.a,pd_HRF_HbR.b,1,1);
        HRF = -amp_Hb./gamma(t/15/fs);
    end
    noised_HRF = Resting + HRF + noise;
    noised_HRF_matrix(i,:) = noised_HRF;
    noise_profile(i,:) = noise;
    HRF_profile(i,:) = HRF;
end
%% save
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
