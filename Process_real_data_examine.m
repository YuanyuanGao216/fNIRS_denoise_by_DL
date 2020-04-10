% Extract qualified data: 0~25s with some Motion artifacts.
% Save the data as Testing data.
% Process data via PCA etc. and save as output HRF and n_MA_left
% Extract motion artifact peak distribution and HRF peak values
% distribution
% what does motion artifact look like in the real data?

clear all
clc
close all
%% add homer path
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% define variables to be calulate
n_MA_HbO_list = [];
n_MA_HbR_list = [];
Real_HbO    = [];
Real_HbR    = [];


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
        d(:,SD.MeasListAct==0) = ones(size(d,1),sum(SD.MeasListAct==0));
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
        fprintf('stim number is %d\n',n_stim)
        
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
            if stim_end - stim_start < 512
                continue
            end
            for Ch = Ch_list
                [n_MA,runstarts,runends] = CalMotionArtifact(tIncChAuto(stim_start:512,Ch));
                if n_MA == 0
                    continue
                end
                if Ch < 36
                    n_MA_HbO_list(end+1) = n_MA;
                else
                    n_MA_HbR_list(end+1) = n_MA;
                end
                if Ch < 36
                    dc_data = squeeze(dc(stim_start:stim_start+512-1,1,Ch)); 
                    Real_HbO(end+1,:)       =   dc_data; 
                    
                else
                    dc_data = squeeze(dc(stim_start:stim_start+512-1,2,Ch-36)); 
                    Real_HbR(end+1,:)       =   dc_data; 
                end
            end
        end
    end
end


