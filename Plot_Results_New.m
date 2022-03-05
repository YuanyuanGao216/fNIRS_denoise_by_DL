%% Figure 1 general idea
clear all
close all
clc
%% add homer path
pathHomer = 'Tools/homer2_src_v2_3_10202017/';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

define_constants
%% Figure 2 Example
% plot_real_data
% plot_sim_data

%% Figure 3A: The number of residual motion artifacts for the simulated testing dataset.
% plot_residual_act
% 
% %%  Figure 3D-I: Example
% 
plot_example
plot_example_newdata
%% MSE table

% make_mse_table



