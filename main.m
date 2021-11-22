%% data simulation
rng(101, 'twister')
% simulate the resting data from real
% SimulateRestingData % sim_Resting
% % characterize noise from real
% Characterize_noise % list
% Simulate motion artifacts
% SimulateMotionArtifacts; % SimulateData; Noise; Resting; n_MA_list;Example
% Do sensitivity analysis for paras in PCA, spline and wavelet on real
% Sensitivity_analysis

%% real data processing
% Use paras above to process real data
% Process_real_data
% Process_real_data_newdata
%% Then train the DAE in python
% pyrunfile('fNIRS_denoise_pytorch_2.py')
%% Use paras above to process simulated data
% Process_test_data
%% Plot results
% Plot_Results_New
