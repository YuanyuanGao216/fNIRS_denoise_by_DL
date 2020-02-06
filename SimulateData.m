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
        rt          =   195/fs; %resting time 24.96s
        pt          =   512/fs; %performance time 65.536s
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
save('Processed_data/simulated_HbO.mat','simulated_HbO')
save('Processed_data/simulated_HbR.mat','simulated_HbR')

