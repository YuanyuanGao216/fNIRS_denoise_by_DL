clear all

%% add homer path
pathHomer   = 'Tools/homer2_src_v2_3_10202017';
oldpath     = cd(pathHomer);
setpaths;
cd(oldpath);

%% define constants
define_constants

STD = 10;

%% load fNIRS file
DataDir = 'Data';
Files = dir(fullfile(DataDir,'*.nirs'));
for file = 1:length(Files) % leave one out
    %% define variables to be calulated
    % amplitude of change
    diff_HbO_list   =   [];
    diff_HbR_list   =   [];
    % duration of change
    duration_HbO_list = [];
    duration_HbR_list = [];
    % density of MA, chances of MA to happen along time
    d_MA_list   =   [];
    
    name = Files(file).name;
    fprintf('fnirs file is %s\n',name);
    fNIRS_data = load([DataDir,'/',name],'-mat');
    %% unify the sampling rate to around 7 Hz
    fNIRS_data  =   Downsample(fNIRS_data);
    %% define variables
    d           =   fNIRS_data.d;
    SD          =   fNIRS_data.SD;
    t           =   fNIRS_data.t;
    tIncMan     =   ones(size(t));
    %% standard processing wo correction
    [dc,SD,tInc]    =   proc_wo_crct(d,SD,t,STD,OD_thred);
    n_Ch            =   size(dc,3);
    %% calculate how many MA detected in this file
    [n_MA,runstarts,runends] = CalMotionArtifact(tInc);
    d_MA_list(end + 1) = n_MA/length(t);
    %% skip the files with no MAs
    if n_MA == 0
        continue
    end
    %% for HbO and HbR
    for bio = 1:2
        %% for each chanel, extract the diffs
        for Ch = 1:n_Ch
            %% skip the noisy channel
            if SD.MeasListAct(Ch) == 0
                continue
            end
            %% loop each MA
            for i = 1:n_MA
                start_point =   runstarts(i) - 1;
                end_point   =   runends(i) - 1;
                MA_HbO      =   squeeze(dc(start_point:end_point,bio,Ch));
                diff_Hb     =   max(MA_HbO)-min(MA_HbO);
                if bio == 1
                    diff_HbO_list(end+1) = diff_Hb;
                    duration_HbO_list(end+1) = end_point - start_point;
                else
                    diff_HbR_list(end+1) = diff_Hb;
                    duration_HbR_list(end+1) = end_point - start_point;
                end
            end
        end
    end
    %%
    % save the diffs and dist. of MA
    data_path = sprintf('Processed_data/leave_%d_out',file);
    mkdir(data_path)
    save([data_path,'/list.mat'],'diff_HbO_list','diff_HbR_list','d_MA_list','duration_HbO_list','duration_HbR_list');
end
