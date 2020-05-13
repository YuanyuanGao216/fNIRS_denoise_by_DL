% Extract qualified data with some Motion artifacts.
% Save the data as real data Real_HbO and Real_HbR.
% Extract motion artifact parameter distributions

%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define variables to be calulate
Real_HbO    = [];
Real_HbR    = [];
diff_HbO_list = [];
diff_HbR_list = [];
n_MA_HbO_list = [];
n_MA_HbR_list = [];
HRF_HbO_list = [];
HRF_HbR_list = [];
Start_HRF_HbO_list = [];
Start_HRF_HbR_list = [];


STD = 10;
%% load fNIRS file
% loop subfolders and files
if ispc
    DataDir = '..\..\Buffalo_study\Raw_Data\Study_#2';
else
    DataDir = '../../Buffalo_study/Raw_Data/Study_#2';
end
Subfolders = dir(DataDir);
for folder = 4:length(Subfolders)
    name = Subfolders(folder).name;
    if strcmp(name,'L8') % L8 dropped out of the study and his data is not complete
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
        tIncMan     =   ones(size(t));
        
        if size(fNIRS_data.s,2) ~= 1
            fNIRS_data.s = sum(fNIRS_data.s,2);
        end
        s           =   fNIRS_data.s;
        %% standard processing wo correction
        SD              =   enPruneChannels(d,SD,tIncMan,[0.01 10],2,[0  45],0);
        dod             =   hmrIntensity2OD(d);
        [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
        dod             =   hmrBandpassFilt(dod,t,0,0.5);
        dc              =   hmrOD2Conc(dod,SD,[6  6]);
        %%
        % select first 65s (512 points) period within trials
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
        for stim_i = 1:2:n_stim
            stim_start   = stimuli_list(stim_i);
            stim_end     = stimuli_list(stim_i+1);
            if stim_end - stim_start < 512
                continue
            end
            for Ch = Ch_list
                [n_MA,runstarts,runends] = CalMotionArtifact(tIncChAuto(stim_start:512,Ch));
                if n_MA == 0
                    continue
                end
                % count how many motion artifacts within one trial and peak
                % height 
                if Ch < 36
                    n_MA_HbO_list(end+1) = n_MA;
                else
                    n_MA_HbR_list(end+1) = n_MA;
                end
                for i = 1:n_MA
                    start_point = runstarts(i)-1;
                    end_point   = runends(i)-1;
                    if Ch <= 36
                        MA_HbO      = squeeze(dc(stim_start+start_point:stim_start+end_point,1,Ch));
                        diff_HbO    = max(MA_HbO)-min(MA_HbO);
                        diff_HbO_list(end+1) = diff_HbO;

                    else
                        MA_HbR      = squeeze(dc(stim_start+start_point:stim_start+end_point,2,Ch-36));
                        diff_HbR    = max(MA_HbR)-min(MA_HbR);
                        diff_HbR_list(end+1) = diff_HbR;
                    end
                end
                if Ch <= 36
                    % save the 0-65s data
                    dc_data = squeeze(dc(stim_start:stim_start+512-1,1,Ch)); 
                    tInc                    =   tIncChAuto(stim_start:stim_start+512-1,Ch);
                    Real_HbO_wo_MA          =   dc_data(find(tInc==1)); 
                    HRF_HbO_list(end+1)     =   max(Real_HbO_wo_MA)-min(Real_HbO_wo_MA);
                    Start_HRF_HbO_list(end+1) = Real_HbO_wo_MA(1);
                else
                    dc_data                 =   squeeze(dc(stim_start:stim_start+512-1,2,Ch-36));
                    tInc                    =   tIncChAuto(stim_start:stim_start+512-1,Ch);
                    Real_HbR_wo_MA          =   dc_data(find(tInc==1)); 
                    HRF_HbR_list(end+1)     =   max(Real_HbR_wo_MA)-min(Real_HbR_wo_MA);
                    Start_HRF_HbR_list(end+1) = Real_HbR_wo_MA(1);

                end
            end
        end
    end
end
%%
pd_diff_HbO = fitdist(diff_HbO_list','gamma');
pd_diff_HbR = fitdist(diff_HbR_list','gamma');
pd_n_HbO = fitdist(n_MA_HbO_list','gamma');
pd_n_HbR = fitdist(n_MA_HbR_list','gamma');
pd_HRF_HbO = fitdist(HRF_HbO_list','gamma');
pd_HRF_HbR = fitdist(HRF_HbR_list','gamma');
pd_start_HRF_HbO = fitdist(Start_HRF_HbO_list','Normal');
pd_start_HRF_HbR = fitdist(Start_HRF_HbR_list','Normal');

save('Processed_data/pds.mat','pd_diff_HbO','pd_diff_HbR','pd_n_HbO','pd_n_HbR','pd_HRF_HbO','pd_HRF_HbR',...
'pd_start_HRF_HbO','pd_start_HRF_HbR');
save('Processed_data/list.mat','diff_HbO_list','diff_HbR_list',...
    'n_MA_HbO_list','n_MA_HbR_list',...
    'HRF_HbO_list','HRF_HbR_list',...
    'Start_HRF_HbO_list','Start_HRF_HbR_list');
figure
histogram(diff_HbO_list);hold on;
title('Dist. of peak/shift distance in HbO')
savefig('Figures/dist. of peakshift distance in HbO.fig')

figure
histogram(diff_HbR_list);hold on;
title('Dist. of peak/shift distance in HbR')
savefig('Figures/dist. of peakshift distance in HbR.fig')

figure
histogram(n_MA_HbO_list);hold on;
title('Dist. of No. of motion artifact for HbO')
savefig('Figures/dist. of No. of motion artifact for HbO.fig')

figure
histogram(n_MA_HbR_list);hold on;
title('Dist. of No. of motion artifact for HbR')
savefig('Figures/dist. of No. of motion artifact for HbR.fig')

figure
histogram(HRF_HbO_list);
title('Dist. of HRF change in HbO')
savefig('Figures/dist. of HRF change in HbO.fig')

figure
histogram(HRF_HbR_list);
title('Dist. of HRF change in HbR')
savefig('Figures/dist. of HRF change in HbO.fig')

figure
histogram(Start_HRF_HbO_list);
title('Dist. of HRF start in HbO')
savefig('Figures/dist. of HRF start in HbO.fig')

figure
histogram(Start_HRF_HbR_list);
title('Dist. of HRF start in HbR')
savefig('Figures/dist. of HRF start in HbO.fig')

