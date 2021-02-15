%% add homer path
pathHomer = 'Tools/homer2_src_v2_3_10202017/';
oldpath = cd(pathHomer);
setpaths;
cd(oldpath);

%% load data
clear all
load('Processed_data/SimulateData.mat','HRF_test_noised')

[m,n] = size(HRF_test_noised);
HbO_noised = HRF_test_noised(1:m/2,:);
HbR_noised = HRF_test_noised(m/2+1:end,:);
%% define variables to be calulate
HbO_Spline  = [];
HbR_Spline  = [];
HbO_Wavelet01 = [];
HbR_Wavelet01 = [];
HbO_Kalman  = [];
HbR_Kalman  = [];
HbO_PCA97   = [];
HbR_PCA97   = [];
HbO_Cbsi    = [];
HbR_Cbsi    = [];

n_Spline = 0;
n_Wavelet01 = 0;
n_Kalman = 0;
n_PCA97 = 0;
n_Cbsi = 0;

define_constants

%% 
SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];
t  = 1/fs_new:1/fs_new:size(HbO_noised,2)/fs_new;
s  = zeros(1,length(t));
s((rt):512:length(t)) = 1;
tIncMan=ones(size(t))';
SD = SD1;
%% Cbsi
T_Cbsi = 0;

for i = 1:m/2
    dc_HbO          =   HbO_noised(i,:);
    dc_HbR          =   HbR_noised(i,:);
    dc              =   [dc_HbO;dc_HbR]';
    
    tic
    [dc_avg,n_MA]   =   proc_Cbsi(dc,s,SD,t,tIncMan,OD_thred,STD);
    T_Cbsi          =   T_Cbsi + toc;
    
    HbO_Cbsi(end+1,:) = dc_avg(:,1)';
    HbR_Cbsi(end+1,:) = dc_avg(:,2)';
    n_PCA97          =   n_PCA97 + n_MA;
end
save('Processed_data/Testing_Cbsi.mat','HbO_Cbsi','HbR_Cbsi','n_Cbsi')

%% PCA97
sigma   =  0.97;
T_PCA97 =   0;

for i = 1:m/2
    dc_HbO          =   HbO_noised(i,:);
    dc_HbR          =   HbR_noised(i,:);
    dc              =   [dc_HbO;dc_HbR]';
    
    tic
    [dc_avg,n_MA]   =   proc_PCA(dc, s, SD, t, tIncMan,STD, OD_thred, sigma);
    T_PCA97         =   T_PCA97 + toc;
    
    HbO_PCA97(end+1,:)  = dc_avg(:,1)';
    HbR_PCA97(end+1,:)  = dc_avg(:,2)';
    n_PCA97             =   n_PCA97 + n_MA;
end
save('Processed_data/Testing_PCA97.mat','HbO_PCA97','HbR_PCA97','n_PCA97')

%% Kalman
T_Kalman = 0;

for i = 1:m/2
    dc_HbO          =   HbO_noised(i,:);
    dc_HbR          =   HbR_noised(i,:);
    dc              =   [dc_HbO;dc_HbR]';
    
    tic
    [dc_avg,n_MA]   =   proc_Kalman(dc,s,SD,t,tIncMan,OD_thred,STD);
    T_Kalman        =   T_Kalman + toc;
    
    HbO_Kalman(end+1,:)  = dc_avg(:,1)';
    HbR_Kalman(end+1,:)  = dc_avg(:,2)';
    n_Kalman             =   n_Kalman + n_MA;
end
save('Processed_data/Testing_Kalman.mat','HbO_Kalman','HbR_Kalman','n_Kalman')
%% Spline
T_Spline =   0;
p       =   0.99;

for i = 1:m/2
    dc_HbO              =   HbO_noised(i,:);
    dc_HbR              =   HbR_noised(i,:);
    dc                  =   [dc_HbO;dc_HbR]';
    
    tic
    [dc_avg,n_MA]       =   proc_Spline(dc, s, SD1, t, tIncMan, STD, OD_thred,p);
    T_Spline            =   T_Spline + toc;
    
    HbO_Spline(end+1,:)  = dc_avg(:,1)';
    HbR_Spline(end+1,:)  = dc_avg(:,2)';
    n_Spline             =   n_Spline + n_MA;
end
save('Processed_data/Testing_Spline.mat','HbO_Spline','HbR_Spline','n_Spline')

%% Wavelet01
T_Wavelet01 =   0;
iqr       =   0.1;

for i = 1:m/2
    dc_HbO              =   HbO_noised(i,:);
    dc_HbR              =   HbR_noised(i,:);
    dc                  =   [dc_HbO;dc_HbR]';
    
    tic
    [dc_avg,n_MA]       =   proc_Wavelet(dc,s,SD1,t,tIncMan,STD,OD_thred,iqr);
    T_Wavelet01         =   T_Wavelet01 + toc;
    
    HbO_Wavelet01(end+1,:)  = dc_avg(:,1)';
    HbR_Wavelet01(end+1,:)  = dc_avg(:,2)';
    n_Wavelet01             =   n_Wavelet01 + n_MA;
end
save('Processed_data/Testing_Wavelet01.mat','HbO_Wavelet01','HbR_Wavelet01','n_Wavelet01')


%% NN count noise
% filepath = 'Processed_data/Test_NN_8layers.mat';
% 
% load(filepath)% File exists.
% Hb_NN = Y_test_predict;
% 
% HbO_NN = Hb_NN(:,1:512);
% HbR_NN = Hb_NN(:,512+1:end);
% HbO_NN = reshape(HbO_NN',[512*5,644]);
% HbR_NN = reshape(HbR_NN',[512*5,644]);
% HbO_NN = HbO_NN';
% HbR_NN = HbR_NN';
% n_NN_HbO = 0;
% n_NN_HbR = 0;
% for i = 1:m/2
%     dc_HbO  =   HbO_NN(i,:);
%     dc_HbR  =   HbR_NN(i,:);
%     dc      =   [dc_HbO;dc_HbR]';
%     dod     =   hmrConc2OD( dc, SD1, ppf );
%     [~,tIncAuto]=   hmrMotionArtifactByChannel(dod,t,SD1,tIncMan,0.5,1,STD,200);
%     
%     [n_MA_HbO,~,~]    =   CalMotionArtifact(tIncAuto(:,1));
%     [n_MA_HbR,~,~]    =   CalMotionArtifact(tIncAuto(:,2));
%     n_NN_HbO = n_NN_HbO + n_MA_HbO;
%     n_NN_HbR = n_NN_HbR + n_MA_HbR;
% end
% save('Processed_data/Testing_NN.mat','n_NN_HbO','n_NN_HbR')
%% output time
time_file = fopen('Processed_data/time.txt','a+');
fprintf(time_file,'Time for Spline is %f\n',T_Spline);
fprintf(time_file,'Time for Wavelet01 is %f\n',T_Wavelet01);
fprintf(time_file,'Time for Kalman is %f\n',T_Kalman);
fprintf(time_file,'Time for PCA97 is %f\n',T_PCA97);
fprintf(time_file,'Time for Cbsi is %f\n',T_Cbsi);
% % T_Spline = 8.637879;
% % T_Wavelet05 = 1111.117784;
% % T_Wavelet35 = 1102.784217;
% % T_Kalman = 55.251534;
% % T_PCA99 = 6.714928;
% % T_PCA50 = 7.102599;
% % T_Cbsi = 0.461004;
% % T_DAE = 0.3288;

% T_list = [T_Spline,T_Wavelet01,T_Kalman,T_PCA99,T_PCA97,T_Cbsi,T_DAE];
% labels = {'Spline','Wavelet01','Kalman','PCA99','PCA79','Cbsi','DAE'};
% figure
% b = bar(T_list,'facecolor',[108, 171, 215]./256,'edgecolor',[108, 171, 215]./256);
% 
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints+30;
% ydata = b(1).YData;
% labels1 = strings(1,8);
% for i = 1:size(ydata,2)
%     string_data = sprintf('%.1f',ydata(i));
%     labels1(i) = string_data;
% end
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% 
% ylabel('Computation time (s)')
% title('Computation time on testing dataset')
% set(gca, 'XTick', 1:size(T_list,2),'fontsize',12)
% set(gca, 'FontName', 'Arial')
% set(gca, 'XTickLabel', labels)
% xtickangle(90)
% set(gcf,'Position',[218   460   330   215]);
% ylim([0 1250])
% xlim([0.5 8.5])
% ax = gca;
% ax.XGrid = 'off';
% ax.YGrid = 'on';
% b.EdgeColor = [200, 14, 80]./255;
% b.FaceColor = 'flat';
% b.CData(9,:) = [200, 14, 80]./255;
% box off