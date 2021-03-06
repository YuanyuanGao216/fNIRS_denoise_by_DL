function [dc_avg,n_MA]         =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD)
global fs_new
dod                     =   hmrConc2OD( dc, SD, [6 6] );
dod                     =   hmrBandpassFilt(dod,t,0,0.5);
dc                      =   hmrOD2Conc(dod,SD,[6  6]);
[dc]                    =   hmrMotionCorrectCbsi(dc,SD,0);
dod                     =   hmrConc2OD(dc,SD,[6  6]);
% [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
tIncAuto            =   hmrMotionArtifact(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
n_MA                    =   count_MA(tIncAuto);