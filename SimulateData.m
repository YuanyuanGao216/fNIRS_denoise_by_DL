% Simulate m samples
% simulated data = Noise + motion artifacts + evoked response
% function output = SimulateData(m)
clear all
clc
close all
seed = 101;

%% noise
% load the fNIRS data
%----------------add homer2 path----------------------------------
pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%-------------select fnirs data path---------------------------
simulated_HbO = [];
simulated_HbR = [];

% loop subfolders and files
DataDir = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2';
Subfolders = dir(DataDir);
for folder = 4:length(Subfolders)
    name = Subfolders(folder).name;
    if strcmp(name,'L8')
        continue
    end
    fprintf('subject is %s\n',name);
    SubDir = [DataDir,'/',name];
    Files = dir(fullfile(SubDir,'*.nirs'));
    for file = 1:length(Files)
        filename = Files(file).name;
        fprintf('fnirs file is %s\n',filename);
        fNIRS_data = load([SubDir,'/',filename],'-mat');
        %% define variables
        d           =   fNIRS_data.d;
        SD          =   fNIRS_data.SD;
        t           =   fNIRS_data.t;
        fs          =   1/(t(2)-t(1));
        rt = 195/fs;%resting time 24.96s
        pt = 512/fs;%performance time 65.536s
        tIncMan     =   ones(size(t));
        %% standard processing wo correction
        SD              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod             =   hmrIntensity2OD(d);
        [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,30,200);
        dod             =   hmrBandpassFilt(dod,t,0,0.5);
        dc              =   hmrOD2Conc(dod,SD,[6  6]);
        %% select period outside trials and fit AR(5) model, simulate new data
        if size(fNIRS_data.s,2) ~= 1
            fNIRS_data.s = sum(fNIRS_data.s,2);
        end
        stimuli_list = find(fNIRS_data.s==1);
        n_stim = length(stimuli_list);
        n_Ch        =   size(dod,2);
        Ch_list     =   1:n_Ch;
        Ch_short    =   [3,6,9,15,20,25,31,36]';
        Ch_short    =   [Ch_short;Ch_short+36];
        Ch_Prune    =   find(SD.MeasListAct==0);
        Ch_list([Ch_short;Ch_Prune]) = [];
        for i = 1:2:n_stim+1
            if i == 1
                rest_start = 1;
                rest_end = stimuli_list(i);
            elseif i == n_stim+1
                rest_start = stimuli_list(i-1);
                rest_end = length(fNIRS_data.s);
            else
                rest_start = stimuli_list(i-1);
                rest_end = stimuli_list(i);
            end
%             rest_time(end+1) = (rest_end-rest_start)/fs;
            if (rest_end-rest_start)/fs < rt
                continue
            else
                for Ch = Ch_list

                    % 0 in tIncAuto means artifact
                    tIncAuto = tIncChAuto(rest_start:rest_end,Ch);
                    dc_rest = dc(rest_start:rest_end,:,:);
                    for j = length(tIncAuto)-fs*rt:-1:1
                        start_point = j;
                        if any(tIncAuto(start_point:start_point+fs*rt)==0)
                            continue
                        else
                            if Ch < 36
                                Resting_HbO = squeeze(dc_rest(start_point:start_point+fs*rt,1,Ch));
                                order = 5;
                                Md = arima(order,0,0);
                                try
                                    EstMd = estimate(Md,Resting_HbO,'Display','off');
                                catch
                                    continue
                                end
                                Constant = EstMd.Constant;
                                AR = cell2mat(EstMd.AR)';
                                Variance = EstMd.Variance;
                                EstMd_HbO = arima('Constant',Constant,'AR',AR,'Variance',Variance); 
                                sim_data = simulate(EstMd_HbO,fs*pt,'NumPaths',1);
                                simulated_HbO(end+1,:) = sim_data;
                            else
                                Resting_HbR = squeeze(dc_rest(start_point:start_point+fs*rt,2,Ch-36));
                                order = 5;
                                Md = arima(order,0,0);
                                try
                                    EstMd = estimate(Md,Resting_HbR,'Display','off');
                                catch
                                    continue
                                end
                                Constant = EstMd.Constant;
                                AR = cell2mat(EstMd.AR)';
                                Variance = EstMd.Variance;
                                EstMd_HbR = arima('Constant',Constant,'AR',AR,'Variance',Variance); 
                                sim_data = simulate(EstMd_HbO,fs*pt,'NumPaths',1);
                                simulated_HbR(end+1,:) = sim_data;
                            end
                            break
                        end
                            
                    end
                end
            end
        end
    end
end
% fprintf('How many resting period? %d\n',size(simulated_HbR,1))
save('Processed_data/simulated_HbO','-mat')
save('Processed_data/simulated_HbR','-mat')

% %------------------------Lap. dist. spikes-----------------------------
% %f(t) = A*exp(-abs(t-t0)/b)
% f = zeros(fs*pt,m);
% for i = 1:m
%     t = 1:fs*pt;
%     n = 10; %how many peaks: 1-10 peaks
%     
%     N = randi(n+1)-1;
%     peak = zeros(fs*pt,1);
%     if N ~= 0
%         for j = 1:N
%             t0 = fs*pt*rand;
%             A_low = 2;A_high = 8;%peak magnitude is from A_low to A_high
%             A = A_low+(A_high-A_low)*rand;%A is from 0.25 to 1
%             b_low = 0;b_high = 1.5;%b is from A_low to A_high
%             b = b_low + rand.*(b_high-b_low);%b is from 0 to 1.5
%             peak = peak + A.*exp(-abs(t-t0)./(b*fs))';
%         end
%     end
%     f(:,i) = peak;
% end
% %------------------------shift----------------------------------------
% shift = zeros(fs*pt,m);
% for i = 1:m
%     n = 3; % how many shifts
%     N = randi(n+1)-1;
%     SHFT = zeros(fs*pt,1);
%     if N ~= 0
%         for j = 1:N
%             transition = round(0.25+(1.5-0.25)*rand);%shift transition time is from 0.25s to 1.5s
%             start_point = round(1+(fs*pt-transition*fs-1-1)*rand);%from 1 to fs*pt-transition*25-1
%             end_point = start_point+transition*fs;
%             DC_shift = (3)+(5-(3))*rand;%from -4 to 4
%             shift_sim = zeros(fs*pt,1);
%             shift_sim(start_point:end_point) = linspace(0,DC_shift,transition*fs+1);
%             shift_sim(end_point:end) = DC_shift;
%             SHFT = SHFT + shift_sim;
%         end
%     end
%     shift(:,i) = SHFT;
% end
% 
% %% evoked response
% HRF_HbO = zeros(fs*pt,m);
% HRF_HbR = zeros(fs*pt,m);
% for i = 1:m
%     t = 1:1:fs*pt;
%     amp_HbO = 15;
%     amp_HbR = -amp_HbO/3;
%     HRF_HbO(:,i) = amp_HbO./gamma(t/5/fs)';
%     HRF_HbR(:,i) = amp_HbR./gamma(t/5/fs)';
% end
% 
% %% synthetic data
% syn_HbO = simulated_HbO + f + shift + HRF_HbO;
% syn_HbR = simulated_HbR + f + shift + HRF_HbR;
% %% save input and output
% %input
% save('datasets\syn_HbO_'+string(m)+'.txt','syn_HbO','-ascii')
% save('datasets\syn_HbR_'+string(m)+'.txt','syn_HbR','-ascii')
% %output
% save('datasets\HRF_HbO_'+string(m)+'.txt','HRF_HbO','-ascii')
% save('datasets\HRF_HbR_'+string(m)+'.txt','HRF_HbR','-ascii')
% %% generate sampling data
% m = 1;
% simulated_HbO = simulate(EstMd_HbO,fs*pt,'NumPaths',m).*1000000;
% simulated_HbR = simulate(EstMd_HbR,fs*pt,'NumPaths',m).*1000000;
% % simulate peaks:
% f = zeros(fs*pt,m);
% for i = 1:m
%     t = 1:fs*pt;
%     n = 10; %how many peaks: 1-10 peaks
%     rng(seed,'twister');
%     N = randi(n+1)-1;
%     peak = zeros(fs*pt,1);
%     if N ~= 0
%         for j = 1:N
%             seed = seed + 1;
%             rng(seed,'twister');
%             t0 = fs*pt*rand;
%             A_low = 2;A_high = 8;%peak magnitude is from A_low to A_high
%             rng(seed,'twister');
%             A = A_low+(A_high-A_low)*rand;%A is from 0.25 to 1
%             b_low = 0;b_high = 1.5;%b is from A_low to A_high
%             rng(seed,'twister');
%             b = b_low + rand.*(b_high-b_low);%b is from 0 to 1.5
%             peak = peak + A.*exp(-abs(t-t0)./(b*fs))';
%         end
%     end
%     f(:,i) = peak;
% end
% plot(f)
% %shift:
% shift = zeros(fs*pt,m);
% for i = 1:m
%     n = 3; % how many shifts
%     rng(seed,'twister');
%     N = randi(n+1);
%     SHFT = zeros(fs*pt,1);
%     if N ~= 0
%         for j = 1:N
%             seed = seed + 1;
%             rng(seed,'twister');
%             transition = round(0.25+(1.5-0.25)*rand);%shift transition time is from 0.25s to 1.5s
%             rng(seed,'twister');
%             start_point = round(1+(fs*pt-transition*fs-1-1)*rand);%from 1 to fs*pt-transition*25-1
%             end_point = start_point+transition*fs;
%             rng(seed,'twister');
%             DC_shift = (3)+(5-(3))*rand;%from -4 to 4
%             shift_sim = zeros(fs*pt,1);
%             shift_sim(start_point:end_point) = linspace(0,DC_shift,transition*fs+1);
%             shift_sim(end_point:end) = DC_shift;
%             SHFT = SHFT + shift_sim;
%         end
%     end
%     shift(:,i) = SHFT;
% end
% figure
% plot(shift)
% HRF_HbO = zeros(fs*pt,m);
% HRF_HbR = zeros(fs*pt,m);
% for i = 1:m
%     t = 1:1:fs*pt;
%     amp_HbO = 15;
%     amp_HbR = -amp_HbO/3;
%     HRF_HbO(:,i) = amp_HbO./gamma(t/5/fs)';
%     HRF_HbR(:,i) = amp_HbR./gamma(t/5/fs)';
% end
% 
% syn_HbO = zeros(fs*pt,4);
% syn_HbR = zeros(fs*pt,4);
% syn_HbO(:,1) = simulated_HbO + HRF_HbO;
% syn_HbO(:,2) = simulated_HbO + f + HRF_HbO;
% syn_HbO(:,3) = simulated_HbO + shift + HRF_HbO;
% syn_HbO(:,4) = simulated_HbO + f + shift + HRF_HbO;
% syn_HbR(:,1) = simulated_HbR + HRF_HbR;
% syn_HbR(:,2) = simulated_HbR - f + HRF_HbR;
% syn_HbR(:,3) = simulated_HbR - shift + HRF_HbR;
% syn_HbR(:,4) = simulated_HbR - f - shift + HRF_HbR;
% title_list = {'No artifact','Spike artifact','Shift artifact','Spike&Shift artifact'};
% figure('Renderer', 'painters', 'Position', [20 20 1200 250])
% hold on
% for i = 1:4
%     subplot(1,4,i)
%     plot(syn_HbO(:,i),'LineWidth',1);
%     axis square
%     title(title_list{i})
%     xlabel('Seconds')
%     ylabel('\muM')
%     xticks([0 fs*10 fs*20])
%     xticklabels({'0','10','20'})
%     ylim([0 25])
%     xlim([0 pt*fs])
% end
% cd('figures')
% savefig('syn_HbO.fig')
% cd('..')
% figure('Renderer', 'painters', 'Position', [20 80 1200 250])
% hold on
% 
% for i = 1:4
%     subplot(1,4,i)
%     plot(syn_HbR(:,i),'LineWidth',1);
%     axis square
%     title(title_list{i})
%     xlabel('Seconds')
%     ylabel('\muM')
%     xticks([0 fs*10 fs*20])
%     xticklabels({'0','10','20'})
%     ylim([-25 0])
%     xlim([0 pt*fs])
% end
% savefig('figures\syn_HbR.fig')
% 
% %input
% save('datasets\syn_HbO_4samples.txt','syn_HbO','-ascii')
% save('datasets\syn_HbR_4samples.txt','syn_HbR','-ascii')
% %output
% HRF_HbO = repmat(HRF_HbO,1,4);
% HRF_HbR = repmat(HRF_HbR,1,4);
% save('datasets\HRF_HbO_4samples.txt','HRF_HbO','-ascii')
% save('datasets\HRF_HbR_4samples.txt','HRF_HbR','-ascii')
% output = 1;