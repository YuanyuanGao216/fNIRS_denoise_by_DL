function [dc,SD,tInc] = proc_wo_crct(d,SD,t,STD)
tIncMan         =   ones(size(t));
SD              =   enPruneChannels(d,SD,tIncMan,[1e+04 1e+07],2,[0  45],0);
dod             =   hmrIntensity2OD(d);
tInc            =   hmrMotionArtifact(dod,t,SD,tIncMan,0.5,1,STD,200);
dod             =   hmrBandpassFilt(dod,t,0,0.5);
dc              =   hmrOD2Conc(dod,SD,[6  6]);