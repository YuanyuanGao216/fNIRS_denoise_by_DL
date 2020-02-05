clear all
close all
clc

[pd_HbO,pd_HbR] = MA_HRF_character;
SimulateData
SimulateMotionArtifacts(pd_HbO,pd_HbR);
BuildRealData
Plot_MAR_results
