function [dc_avg,n_MA] =   proc_PCA(dc, s,SD, t, tIncMan,STD, sigma)
global fs_new
dod                     =   hmrConc2OD( dc, SD, [6 6] );
[dod,~,~,~,~]           =   hmrMotionCorrectPCArecurse(dod,t,SD,tIncMan,0.5,1,STD,200,sigma,5);
[~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,200);
dod                     =   hmrBandpassFilt(dod,t,0,0.5);
dc                      =   hmrOD2Conc(dod,SD,[6  6]);
[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
n_MA                    =   count_no_correction(tIncAuto);