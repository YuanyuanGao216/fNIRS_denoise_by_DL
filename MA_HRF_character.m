% Extract motion artifact peak distribution and HRF peak values
% distribution
% what does motion artifact look like in the real data?
function [pd_HbO,pd_HbR] = MA_HRF_character()

clear all
clc
close all
%% add homer path
pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% load fNIRS file
diff_HbO_list = [];
diff_HbR_list = [];
n_MA_HbO_list = [];
n_MA_HbR_list = [];
% loop subfolders and files
DataDir = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2';
Subfolders = dir(DataDir);
for i = 4:length(Subfolders)
    name = Subfolders(i).name;
    if strcmp(name,'L8')
        continue
    end
    fprintf('subject is %s\n',name);
    SubDir = [DataDir,'/',name];
    Files = dir(fullfile(SubDir,'*.nirs'));
    for j = 1:length(Files)
        filename = Files(j).name;
        fprintf('fnirs file is %s\n',filename);
        fNIRS_data = load([SubDir,'/',filename],'-mat');
        %% define variables
        d           =   fNIRS_data.d;
        SD          =   fNIRS_data.SD;
        t           =   fNIRS_data.t;
        tIncMan     =   ones(size(t));
        s           =   fNIRS_data.s;

        %% standard processing wo correction
        SD              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);

        dod             =   hmrIntensity2OD(d);

        [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,30,200);

        dod             =   hmrBandpassFilt(dod,t,0,0.5);

        dc              =   hmrOD2Conc(dod,SD,[6  6]);
        %%
        % select period within trials
        if size(fNIRS_data.s,2) ~= 1
            fNIRS_data.s = sum(fNIRS_data.s,2);
        end
        stimuli_list = find(fNIRS_data.s==1);
        n_stim = length(stimuli_list);

        % 0 in tIncAuto means artifact
        n_Ch        =   size(dod,2);
        Ch_list     =   1:n_Ch;
        Ch_short    =   [3,6,9,15,20,25,31,36]';
        Ch_short    =   [Ch_short;Ch_short+36];
        Ch_Prune    =   find(SD.MeasListAct==0);
        Ch_list([Ch_short;Ch_Prune]) = [];
        for i = 1:2:n_stim
            stim_start   = stimuli_list(i);
            stim_end     = stimuli_list(i+1);
            for Ch = Ch_list
                tIncAuto = tIncChAuto(stim_start:stim_end,Ch);
                transitions = diff([0; tIncAuto == 0; 0]);
                runstarts = find(transitions == 1);
                runends = find(transitions == -1);
                if ~isempty(runstarts)
                    n_MA = length(runstarts);
                    if Ch < 36
                        n_MA_HbO_list(end+1) = n_MA;
                    else
                        n_MA_HbR_list(end+1) = n_MA;
                    end
                    for i = 1:n_MA
                        start_point = runstarts(i)-1;
                        end_point = runends(i)-1;
                        if Ch < 36
                            MA_HbO = squeeze(dc(stim_start+start_point:stim_start+end_point,1,Ch));
                            diff_HbO = max(MA_HbO)-min(MA_HbO);
                            diff_HbO_list(end+1) = diff_HbO;

                        else
                            MA_HbR = squeeze(dc(stim_start+start_point:stim_start+end_point,2,Ch-36));
                            diff_HbR = max(MA_HbR)-min(MA_HbR);
                            diff_HbR_list(end+1) = diff_HbR;
                        end
                    end
                end
            end
        end
    end
end
%%
pd_HbO = fitdist(diff_HbO_list','gamma')
pd_HbR = fitdist(diff_HbR_list','gamma')

f_HbO = figure;
histogram(diff_HbO_list);hold on;
% simu = gamrnd(pd_HbO.a,pd_HbO.b,1500,1);
title('HbO')
%%
figure
histogram(diff_HbR_list);hold on;
title('HbR')

figure
histogram(n_MA_HbO_list);hold on;
title('HbO')

figure
histogram(n_MA_HbR_list);hold on;
title('HbR')
