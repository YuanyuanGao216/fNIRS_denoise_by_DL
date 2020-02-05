% pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';;
% oldpath = cd(pathHomer);
% setpaths;
% cd(oldpath);
% cd('/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2');
% EasyNIRS

Mdl = arima('AR',{0.75,0.15},'SAR',{0.9,-0.75,0.5},...
    'SARLags',[12,24,36],'MA',-0.5,'Constant',2,...
    'Variance',1);
rng(1);
y = simulate(Mdl,1000);
figure
parcorr(y)
figure
parcorr(y,40)