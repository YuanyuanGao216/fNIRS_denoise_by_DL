% pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';;
% oldpath = cd(pathHomer);
% setpaths;
% cd(oldpath);
% cd('/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2');
% EasyNIRS
file = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2/NIRS-2019-08-10_005.nirs';
fNIRS_data = load(file,'-mat');
d           =   fNIRS_data.d;
SD          =   fNIRS_data.SD;
t           =   fNIRS_data.t;
tIncMan     =   ones(size(t));
s           =   fNIRS_data.s;
dod             =   hmrIntensity2OD(d);
[~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,30,200);


fs = 7.8125;
SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
d_1 = d(:,[2 2+36]);
dod1             =   hmrIntensity2OD(d_1);
[~,tIncChAuto2] = hmrMotionArtifactByChannel(dod1, t, SD1, ones(size(d_1)), 0.5, 1, 30, 200);
% fs = 7.8125;
% SD1.MeasList = [1,1,1,1];
% SD1.MeasListAct = [1];
% SD1.Lambda = [760];
% d_1 = d(:,[2+36]);
% dod1             =   hmrIntensity2OD(d_1);
% [~,tIncChAuto2] = hmrMotionArtifactByChannel(dod1, t, SD1, ones(size(d_1)), 0.5, 1, 30, 200);

figure
plot(tIncChAuto(:,2+36),'b-');hold on;
plot(tIncChAuto2(:,1),'r-');hold on;
sum(tIncChAuto(:,2+36)) == sum(tIncChAuto2(:,1))