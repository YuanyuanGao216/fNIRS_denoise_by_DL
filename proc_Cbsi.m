function [dc_avg,n_MA]         =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD)
global fs_new
dod                     =   hmrConc2OD_modified( dc, SD, [1 1] );
dod                     =   hmrBandpassFilt(dod,t,0,0.5);
dc                      =   hmrOD2Conc_modified(dod, SD, [1  1]);
[dc]                    =   hmrMotionCorrectCbsi(dc,SD,0);
% dod                     =   hmrConc2OD(dc,SD,[6  6]);
dod                      =   hmrConc2OD_modified(dc,SD,[1  1]);% here make the conc unit um*mm
% [~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
tIncAuto            =   hmrMotionArtifact(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
n_MA                    =   count_MA(tIncAuto);