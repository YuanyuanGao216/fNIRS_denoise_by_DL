clear all
close all
clc

[pd_HbO,pd_HbR] = MA_HRF_character;
SimulateData
SimulateMotionArtifacts;
BuildRealData
Plot_Results
