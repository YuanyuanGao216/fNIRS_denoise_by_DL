function [dc_avg,n_MA]       =   proc_Kalman(dc,s,SD,t,tIncMan,OD_thred,STD)
global fs_new
dod                     =   hmrConc2OD( dc, SD, [6 6] );
y                       =   dod;
oscFreq                 =   [0,0.01,0.001,0.0001];
xo                      =   ones(1,length(oscFreq)+1)*y(1,1);
Po                      =   ones(1,length(oscFreq)+1)*(y(1,1)^2);
Qo                      =   zeros(1,length(oscFreq)+1);
hrfParam                =   [2 2];
[~, ~,dod,~,~,~]        =   hmrKalman2( y, s', t, xo, Po, Qo, 'box', hrfParam, oscFreq );
[~,tIncAuto]            =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,STD,OD_thred);
dod                     =   hmrBandpassFilt(dod,t,0,0.5);
dc                      =   hmrOD2Conc(dod,SD,[6  6]);
[dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
n_MA                    =   count_MA(tIncAuto);