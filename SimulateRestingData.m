close all
clear all
%% define constants
define_constants
%% ----------------add homer2 path----------------------------------
pathHomer = '../../Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);
%% variables to be calculated
simulated_HbO = [];
simulated_HbR = [];

%% loop subfolders and files
DataDir = '../New Datasets/05/motion_artifacts_1';
Files = dir(fullfile(DataDir,'*.nirs'));
figure_flag = 0;
for file = 1:length(Files) - 1 % save the last one for testing
    name = Files(file).name;
    fprintf('fnirs file is %s\n',name);
    fNIRS_data = load([DataDir,'/',name],'-mat');
    
    %% unify the sampling rate to around 7.8 Hz
    fNIRS_data  =   Downsample(fNIRS_data);
    
    %% define variables
    d           =   fNIRS_data.d;
    SD          =   fNIRS_data.SD;
    t           =   fNIRS_data.t;
    fs          =   1 / (t(2) - t(1));
    fprintf('fs is %f\n',fs);
    
    %% standard processing wo correction
    [dc,SD,tInc] = proc_wo_crct(d,SD,t,STD, OD_thred);
    
    %% visualize the data from one channel and the detected Montion Artifacts
    channel = 2;
    
    %% select period without MA and fit AR(5) model, simulate new data
    n_Ch        =   size(dc,3);
    for bio = 1:2
        for Ch = 1:n_Ch
            % exclude the noisy channel
            if SD.MeasListAct(Ch) == 0
                continue
            end
            % scan along the time line
            period = 512;
            for start_point = 1:1:(length(tInc) - period)
                t_span = start_point : start_point + period -1;
                % if contains noise, skip
                if any(tInc(t_span)==0)
                    continue
                end
                % try 10 times AR simulation
                for n = 1:10
                    Resting_HbO = squeeze(dc(t_span,bio,Ch));
                    [sim_data_HbO,state] = AR_sim(Resting_HbO);
                    % if the amplitude of sim is 30 times larger, skip
                    if abs(max(sim_data_HbO)) > abs(max(Resting_HbO)) * 30 || state == 1
                        continue
                    end
                    % plot an example
                    if figure_flag == 0 && Ch == channel
                        plot_real_data(t,dc,tInc,channel,t_span,Resting_HbO,sim_data_HbO)
                        save('Processed_data/plot_Resting.mat','t','dc','tInc','channel','t_span','Resting_HbO','sim_data_HbO');
                    end
                    % if simulate successfully, record in simulated_HbO and _HbR
                    if bio == 1
                        if figure_flag == 0
                            figure_flag = 1;
                            loc_example = size(simulated_HbO,1) + 1;
                        end
                        simulated_HbO(end+1,:) = sim_data_HbO;
                    else
                        simulated_HbR(end+1,:) = sim_data_HbO;
                    end
                    
                    break
                end
            end
        end
    end
end
fprintf('%d samples in HbO\n', size(simulated_HbO,1))
fprintf('%d samples in HbR\n', size(simulated_HbR,1))

save('Processed_data/sim_Resting.mat','simulated_HbO','simulated_HbR', 'loc_example')


