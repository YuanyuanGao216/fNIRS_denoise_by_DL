%% data simulation
% simulate the resting data from real
SimulateRestingData % sim_Resting
% characterize noise from real
Characterize_noise % list
% Simulate motion artifacts
SimulateMotionArtifacts; % SimulateData; Noise; Resting; n_MA_list;Example
% Do sensitivity analysis for paras in PCA, spline and wavelet on real
Sensitivity_analysis

%% real data processing
% Use paras above to process real data
Process_real_data

% Use paras above to process simulated data
Process_test_data

%% Then train the DAE in python
%% Plot results
Plot_Results_New % real data
