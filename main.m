%% data simulation
% simulate the resting data from real
SimulateRestingData
% characterize noise from real
Characterize_noise
% Simulate motion artifacts
SimulateMotionArtifacts;
% Do sensitivity analysis for paras in PCA, spline and wavelet on real
Sensitivity_analysis

%% real data processing
% Use paras above to process real data
Process_real_data
% Use paras above to process simulated data
Process_test_data

%% Then train the DAE in python
% Plot results
Plot_test_data % simulated data
Plot_Results % real data
Plot_loss
