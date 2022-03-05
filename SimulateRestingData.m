close all
clear all
%% define constants
define_constants
%% ----------------add homer2 path----------------------------------
pathHomer = 'Tools/homer2_src_v2_3_10202017';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

%% loop subfolders and files
DataDir = 'Data';
Files = dir(fullfile(DataDir,'*.nirs'));
figure_flag = 0;
% for file_leftout = 1:length(Files) % leave one out
for file_leftout = 8:8 % leave one out
    %% variables to be calculated
    simulated_HbO = [];
    simulated_HbR = [];
    stop_sign = 0;
    
    fprintf('leaving %d file out.........\n',file_leftout)
    
    name = Files(file_leftout).name;
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

    %% select period without MA and fit AR(5) model, simulate new data
    n_Ch        =   size(dc,3);
    for Ch = 1:n_Ch
        % exclude the noisy channel
        if SD.MeasListAct(Ch) == 0
%             fprintf('skip ch\n')
            continue
        end
        % scan along the time line
        period = 75;
        for start_point = 1:1:(length(tInc) - period)
            t_span = start_point : start_point + period -1;
            % if contains noise, skip
            if any(tInc(t_span)==0)
%                 fprintf('skip t span\n')
                continue
            end
            % try 10 times AR simulation
            for n = 1:10
                Resting_HbO = squeeze(dc(t_span,1,Ch));
                Resting_HbR = squeeze(dc(t_span,2,Ch));
                [sim_data_HbO,state1] = AR_sim(Resting_HbO);
                [sim_data_HbR,state2] = AR_sim(Resting_HbR);
%                 sim_data_HbO = zeros(1, 512*5);
%                 sim_data_HbR = zeros(1, 512*5);
%                 state1 = 0;
%                 state2 = 0;
                % if the amplitude of sim is 30 times larger, skip
                if state1 == 1 || abs(max(sim_data_HbO)) > abs(max(Resting_HbO)) * 30
                    continue
                end
                if state2 == 1 || abs(max(sim_data_HbR)) > abs(max(Resting_HbR)) * 30 
                    continue
                end
                % plot an example
                if file_leftout == 2 && Ch == 1 && figure_flag == 0
                    plot_real_data(t,dc,tInc,Ch,t_span,Resting_HbO,sim_data_HbO)
                    figure_flag = 1;
                    loc_example = size(simulated_HbO,1) + 1;
                    save('Processed_data/plot_Resting.mat','t','dc','tInc','Ch','t_span','Resting_HbO','sim_data_HbO','loc_example');
                end

                % if the amplitude is ok, we save the data and stop the simulation
                
                simulated_HbO(end+1,:) = sim_data_HbO;
                simulated_HbR(end+1,:) = sim_data_HbR;
                break
            end
            
            if size(simulated_HbO,1) >= 2000 || size(simulated_HbR,1) >= 2000
                stop_sign = 1;
                break
            end
        end
        if stop_sign == 1
            break
        end
    end
    

    fprintf('%d samples in HbO\n', size(simulated_HbO,1))
    fprintf('%d samples in HbR\n', size(simulated_HbR,1))
    data_path = sprintf('Processed_data/leave_%d_out',file_leftout);
    mkdir(data_path)
    if exist('loc_example','var')
        save([data_path,'/sim_Resting.mat'],'simulated_HbO','simulated_HbR', 'loc_example')
    else
        save([data_path,'/sim_Resting.mat'],'simulated_HbO','simulated_HbR')
    end
end