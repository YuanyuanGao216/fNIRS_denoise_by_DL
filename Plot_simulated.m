close all

load('Processed_data/pds.mat')
fs = 7.8125;
pt = 512/fs;
t = 1:fs*pt;
seed = 101;
rng(seed);
% resting data
load('Processed_data/simulated_HbO.mat','simulated_HbO')

resting_data = simulated_HbO(3,:);

figure
subplot(5,1,1)
plot(resting_data)
ylim([-4.5 4.5]*1e-6)
title('resting data')

% spike noise
t0 = fs*pt*rand;
A = gamrnd(pd_diff_HbO.a,pd_diff_HbO.b,1,1);
b_low = 0;b_high = 1.5;%b is from A_low to A_high
b = b_low + rand.*(b_high-b_low);%b is from 0 to 1.5
spike_noise = A.*exp(-abs(t-t0)./(b*fs));

subplot(5,1,2)
plot(spike_noise)
title('spike noise')
ylim([-4.5 4.5]*1e-6)

% shift noise
transition = round(0.25+(1.5-0.25)*rand);%shift transition time is from 0.25s to 1.5s
start_point = 400;%from 1 to fs*pt-transition*25-1
end_point = start_point+transition*fs;
DC_shift = 9e-07;
shift_sim = zeros(fs*pt,1);
shift_sim(start_point:end_point) = linspace(0,DC_shift,transition*fs+1);
shift_sim(end_point:end) = DC_shift;
shift_noise = shift_sim';

subplot(5,1,3)
plot(shift_noise)
title('shift noise')
ylim([-4.5 4.5]*1e-6)

% HRF
amp_Hb = 2.9000e-06;
HRF = amp_Hb./gamma(t/15/fs);

subplot(5,1,4)
plot(HRF)
title('HRF')
ylim([-4.5 4.5]*1e-6)

% noisy HRF
noised_HRF = resting_data + HRF + shift_noise + spike_noise;

subplot(5,1,5)
plot(noised_HRF)
title('noised_HRF')
ylim([-4.5 4.5]*1e-6)

set(gcf,'Position',[538    92   156   611])

saveas(gcf,'Figures/Simulated','svg')