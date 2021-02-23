%% Figure 1 general idea

%% Figure 2 Example
plot_real_data
plot_sim_data

%% Figure 4
load('Processed_data/Testing_Spline.mat')
load('Processed_data/Testing_Wavelet01.mat')
load('Processed_data/Testing_Kalman.mat')
load('Processed_data/Testing_PCA99.mat')
load('Processed_data/Testing_PCA79.mat')
load('Processed_data/Testing_Cbsi.mat')
load('Processed_data/Testing_NN.mat')

load('Processed_data/peak_n_list.mat','peak_n_list')
load('Processed_data/shift_n_list.mat','shift_n_list')
load('Processed_data/random_profile.mat','p')

m           =   size(peak_n_list,1);
n_train     =   round((m/2)*0.8);
n_val       =   round((m/2)*0.1);
test_idx    =   [p(n_train+n_val+1:end),p(n_train+n_val+1:end)+m/2];

peak_n_list_test    =   peak_n_list(test_idx,:);
shift_n_list_test   =   shift_n_list(test_idx,:);
n_test              =   size(peak_n_list_test,1)/2;
n_MA_total_HbO      =   sum(peak_n_list_test(1:n_test,:))+sum(shift_n_list_test(1:n_test,:));
n_MA_total_HbR      =   sum(peak_n_list_test(n_test+1:end,:))+sum(shift_n_list_test(n_test+1:end,:));

MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet01_HbO,...
    n_Kalman_HbO,n_PCA99_HbO,n_PCA79_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet01_HbR,...
    n_Kalman_HbR,n_PCA99_HbR,n_PCA79_HbR,n_Cbsi_HbO,n_NN_HbR];

MA_list_new = MA_list(1,:);
MA_list_percetage = MA_list_new./MA_list_new(1);

figure
b = bar(MA_list_percetage,'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);
labels = {'No correction','Spline','Wavelet01','Kalman','PCA99','PCA79','Cbsi','DAE'};
ylabel({'Residual motion artifacts'})
set(gca, 'XTick', 1:size(MA_list_new,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
yticks([0 1])
yticklabels({'0','100%'})
ylim([0 1.5])
xlim([0 10])

xtips1  =   b(1).XEndPoints;
ytips1  =   b(1).YEndPoints;
per_label = b(1).YData;

for i = 1:length(per_label)
    x = per_label(i)*100;
    label = sprintf('%.0f%%',x);
    text(xtips1(i),ytips1(i),label,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end
% labels1 =   [string(b(1).YData*100),'%'];

set(gcf,'Position',[3   490   330   215]);
box off
fprintf('%d\n',MA_list(1,end))
MA_list(1,:)
%% Figure 4.2
load('Processed_data/Process_real_data.mat',...
    'HRFs',...
    'dc',...
    'dc_avg_no_correction',...
    'dc_avg_PCA99',...
    'dc_avg_PCA79',...
    'dc_avg_Spline',...
    'dc_avg_Wavelet01',...
    'dc_avg_Kalman',...
    'dc_avg_Cbsi',...
    'n_MA_no_correction',...
    'n_MA_PCA99',...
    'n_MA_PCA79',...
    'n_MA_Spline',...
    'n_MA_Wavelet01',...
    'n_MA_Kalman',...
    'n_MA_Cbsi')

filepath = 'Processed_data/Real_NN_8layers_save.mat';
load(filepath)% File exists.
Hb_NN = Y_real;

SD1.MeasList = [1,1,1,1;1,1,1,2];
SD1.MeasListAct = [1 1];
SD1.Lambda = [760;850];
SD1.SrcPos = [-2.9017 10.2470 -0.4494];
SD1.DetPos = [-4.5144 9.0228 -1.6928];
ppf = [6,6];



n_NN_HbO = 0;
n_NN_HbR = 0;
for i = 1:size(HbO_NN,1)
    dc_HbO = HbO_NN(i,:);
    dc_HbR = HbR_NN(i,:);
    dc = [dc_HbO;dc_HbR]';
    dod = hmrConc2OD( dc, SD1, ppf );
    [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,1));
    if n_MA ~= 0
        fprintf('%d\n',i)
    end
    n_NN_HbO = n_NN_HbO + n_MA;
    [n_MA,~,~] = CalMotionArtifact(tIncChAuto(:,2));
    n_NN_HbR = n_NN_HbR + n_MA;
end


MA_list = [n_MA_total_HbO,n_Spline_HbO,n_Wavelet05_HbO,n_Wavelet35_HbO,...
    n_Kalman_HbO,n_PCA99_HbO,n_PCA50_HbO,n_Cbsi_HbO,n_NN_HbO;...
    n_MA_total_HbR,n_Spline_HbR,n_Wavelet05_HbR,n_Wavelet35_HbR,...
    n_Kalman_HbR,n_PCA99_HbR,n_PCA50_HbR,n_Cbsi_HbO,n_NN_HbR];


figure
b = bar(MA_list(1,:),'facecolor',[108, 171, 215]./256,'edgecolor',[1 1 1]);

ylabel('No. of Motion Artifacts')
set(gca, 'XTick', 1:size(MA_list,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
ylim([0 600])
xlim([0 10])

xtips1  =   b(1).XEndPoints;
ytips1  =   b(1).YEndPoints;
labels1 =   string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
set(gcf,'Position',[3   490   330   215]);
box off
fprintf('%d\n',MA_list(1,end))
MA_list(1,:)