% pathHomer = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Tools/homer2_src_v2_3_10202017';;
% oldpath = cd(pathHomer);
% setpaths;
% cd(oldpath);
% cd('/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2');
% EasyNIRS
% file = '/Users/gaoyuanyuan/Dropbox/Cemsim/Studies/Buffalo_study/Raw_Data/Study_#2/L2/NIRS-2019-08-10_005.nirs';
% fNIRS_data = load(file,'-mat');
% d           =   fNIRS_data.d;
% SD          =   fNIRS_data.SD;
% t           =   fNIRS_data.t;
% tIncMan     =   ones(size(t));
% s           =   fNIRS_data.s;
% dod             =   hmrIntensity2OD(d);
% [~,tIncChAuto]  =   hmrMotionArtifactByChannel(dod,t,SD,tIncMan,0.5,1,30,200);
% 
% 
% fs = 7.8125;
% SD1.MeasList = [1,1,1,1;1,1,1,2];
% SD1.MeasListAct = [1 1];
% SD1.Lambda = [760;850];
% d_1 = d(:,[2 2+36]);
% dod1             =   hmrIntensity2OD(d_1);
% [~,tIncChAuto2] = hmrMotionArtifactByChannel(dod1, t, SD1, ones(size(d_1)), 0.5, 1, 30, 200);
% % fs = 7.8125;
% % SD1.MeasList = [1,1,1,1];
% % SD1.MeasListAct = [1];
% % SD1.Lambda = [760];
% % d_1 = d(:,[2+36]);
% % dod1             =   hmrIntensity2OD(d_1);
% % [~,tIncChAuto2] = hmrMotionArtifactByChannel(dod1, t, SD1, ones(size(d_1)), 0.5, 1, 30, 200);
% 
% figure
% plot(tIncChAuto(:,2+36),'b-');hold on;
% plot(tIncChAuto2(:,1),'r-');hold on;
% sum(tIncChAuto(:,2+36)) == sum(tIncChAuto2(:,1))
piece = input('Which piece you want to show? ');

figure('Renderer', 'painters','Position',[10 10 1200 250]);
subplot(1,2,1)
plot(Real_HbO(piece,1:4*fs),'k-','DisplayName','No correction');hold on;
plot(HbO_Spline(piece,1:4*fs),'b-','DisplayName','Spline');hold on;
plot(HbO_Wavelet(piece,1:4*fs),'g-','DisplayName','Wavelet');hold on;
plot(HbO_Kalman(piece,1:4*fs),'y-','DisplayName','Kalman');hold on;
plot(HbO_PCA97(piece,1:4*fs),'m-','DisplayName','PCA97');hold on;
plot(HbO_PCA80(piece,1:4*fs),'c-','DisplayName','PCA80');hold on;
plot(HbO_Cbsi(piece,1:4*fs),'Color',[153, 102, 51]./255,'DisplayName','Cbsi');hold on;
plot(HbO_NN(piece,1:4*fs),'r-','DisplayName','DAE');hold on;
ylabel('/delta HbO (/muMol)')
fs = 7.8125;
% xticks([0 20 40 60]*fs)
% xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show
subplot(1,2,2)
plot(Real_HbR(piece,1:4*fs),'k-','DisplayName','No correction');hold on;
plot(HbR_Spline(piece,1:4*fs),'b-','DisplayName','Spline');hold on;
plot(HbR_Wavelet(piece,1:4*fs),'g-','DisplayName','Wavelet');hold on;
plot(HbR_Kalman(piece,1:4*fs),'y-','DisplayName','Kalman');hold on;
plot(HbR_PCA97(piece,1:4*fs),'m-','DisplayName','PCA97');hold on;
plot(HbR_PCA80(piece,1:4*fs),'c-','DisplayName','PCA80');hold on;
plot(HbR_Cbsi(piece,1:4*fs),'Color',[153, 102, 51]./255,'DisplayName','Cbsi');hold on;
plot(HbR_NN(piece,1:4*fs),'r-','DisplayName','DAE');hold on;
ylabel('/delta HbR (/muMol)')
fs = 7.8125;
% xticks([0 20 40 60]*fs)
% xticklabels({'0s','20s','40s','60s'})
legend('Location','northeastoutside')
legend show