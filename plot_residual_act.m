%% 3A residual MA boxplot with error bar
% now we have 10 testing datasetsDataDir = 'Processed_data';
DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
MA_list_percentage = zeros(6,length(subfolders));
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Spline.mat');
    load(test_path)
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Wavelet01.mat');
    load(test_path)
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Kalman.mat');
    load(test_path)
%     test_path = fullfile(DataDir, subfolders(subfolder).name), 'Testing_PCA97.mat');
%     load(test_path)
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Cbsi.mat');
    load(test_path)
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'Testing_NN.mat');
    load(test_path)
    
    test_path = fullfile(DataDir, subfolders(subfolder).name, 'n_MA_list.mat');
    load(test_path,'peak_n_list','shift_n_list','p')
    peak_n_list = peak_n_list';
    shift_n_list = shift_n_list';
    
    n_sample    =   size(peak_n_list,1);
    n_train     =   round(n_sample*0.8);
    n_val       =   round(n_sample*0.1);
    train_idx   =   p(1:n_train);
    val_idx     =   p(n_train+1:n_train+n_val);
    test_idx    =   p(n_train+n_val+1:end);

    peak_n_list_test    =   peak_n_list(test_idx,:);
    shift_n_list_test   =   shift_n_list(test_idx,:);

    n_MA_no_correction      =   sum(peak_n_list_test)+sum(shift_n_list_test);
    
    MA_list = [n_MA_no_correction,...
            n_Spline,...
            n_Wavelet01,...
            n_Kalman,...
            n_Cbsi,...
            n_NN];
    MA_list_percentage(:,subfolder) = MA_list./MA_list(1);
end

figure
mean_MA = mean(MA_list_percentage,2);
std_MA = std(MA_list_percentage,[],2);
b = bar(1:6, mean_MA, 'facecolor', [108, 171, 215]./256, 'edgecolor', [1 1 1]);
hold on
errorbar(1:6, mean_MA, std_MA, 'color', [108, 171, 215]./256, 'linestyle','none' )
hold off
labels = {'No correction','Spline','Wavelet','Kalman','Cbsi','DAE'};
ylabel({'Residual motion artifacts'})
set(gca, 'XTick', 1:length(MA_list),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
yticks([0 1])
yticklabels({'0','100%'})
ylim([0 1.5])
xlim([0 7])

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
savefig(fullfile('Figures','3A_boxplot_MA_testing.fig'))


fprintf('For 3A sig test\n')
% sig test
for i = 2:size(MA_list_percentage,1)
    x1 = MA_list_percentage(1,:);
    x2 = MA_list_percentage(i,:);
    h = kstest(x1 - x2);
    if h == 1
        p = signtest(x1 - x2);
    else
        p = ttest(x1-x2);
    end
    fprintf('for %s:, h = %d; p = %.3f\n', labels{i}, h, p)
end
%% Figure 3B: The number of residual motion artifacts for the real dataset.

% save No. of MAs for each files n_files X [n_MA_no_correction,dc_avg_PCA97,n_MA_Spline99,n_MA_Wavelet01,]
load('Processed_data/Process_real_data.mat','Proc_data','MA_matrix')

subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
n_NN = zeros(size(MA_matrix,1),1);
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Real_NN_8layers_act.mat');
    load(filepath)% File exists.
    Hb_NN = Y_real_act;
%     filepath = fullfile(DataDir, subfolders(subfolder).name, 'number_array.mat');
%     load(filepath)

    SD1.MeasList = [1,1,1,1;1,1,1,2];
    SD1.MeasListAct = [1 1];
    SD1.Lambda = [760;850];
    SD1.SrcPos = [-2.9017 10.2470 -0.4494];
    SD1.DetPos = [-4.5144 9.0228 -1.6928];
    ppf = [1,1];

    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    n = 0;
    for i = 1:size(HbO_NN,1)
        dc_HbO = HbO_NN(i,:);
        dc_HbR = HbR_NN(i,:);
        dc = [dc_HbO;dc_HbR]';
        dod = hmrConc2OD_modified( dc, SD1, ppf );
        [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs_new, SD1, ones(size(dod,1)), 0.5, 1, STD, 200);
        n_MA = count_MA(tIncChAuto);
        n = n + n_MA;
    end
    n_NN(subfolder) = n;
end

MA_matrix = [MA_matrix, n_NN];

figure
errorbar(1:size(MA_matrix,2),mean(MA_matrix,1),std(MA_matrix,[],1), 'color','b','marker','o','MarkerFaceColor','b' )

ylabel('Residual motion artifacts')
labels = {'No correction','Spline','Wavelet','Kalman','PCA97','CBSI','DAE'};
set(gca, 'XTick', 1:size(MA_matrix,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
% ylim([0 40])
xlim([0.5 8.5])

set(gcf,'Position',[3   431   390   274]);
box off

fprintf('For 3B sig test\n')
% sig test
for i = 2:size(MA_matrix,2)
    x1 = MA_matrix(:,1);
    x2 = MA_matrix(:,i);
    h = kstest(x1 - x2);
    if h == 1
        p = signtest(x1 - x2);
    else
        p = ttest(x1-x2);
    end
    fprintf('for %s:, h = %d; p = %.3f\n', labels{i}, h, p)
end

%% Figure 3C: Act vs. No act.
n_channels = 14;
n_files = length(Proc_data);

Real_matrix = zeros(n_channels * n_files, 7);
Real_matrix_act = zeros(n_channels * n_files, 7);

for i = 1:n_files
    index = (i-1)*n_channels+1 : i*n_channels;
    j = 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_no_crct(1:512,1,:),1)); j = j + 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_Spline(1:512,1,:),1)); j = j + 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_Wavelet(1:512,1,:),1)); j = j + 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_Kalman(1:512,1,:),1)); j = j + 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_PCA97(1:512,1,:),1)); j = j + 1;
    Real_matrix(index,j) = squeeze(mean(Proc_data(i).dc_Cbsi(1:512,1,:),1)); 

    j = 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_no_crct(1:512,1,:),1)); j = j + 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_Spline(1:512,1,:),1)); j = j + 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_Wavelet(1:512,1,:),1)); j = j + 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_Kalman(1:512,1,:),1)); j = j + 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_PCA97(1:512,1,:),1)); j = j + 1;
    Real_matrix_act(index,j) = squeeze(mean(Proc_data(i).dc_act_Cbsi(1:512,1,:),1));
end

subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
MA_list_percentage = zeros(6,length(subfolders));
for subfolder = 1:length(subfolders)
    fprintf('subfolder is %d\n', subfolder)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Real_NN_8layers.mat');
    load(filepath)
    
    Hb_NN = Y_real;

    HbO_NN = mean(Hb_NN(:, 1:512),2);
    HbO_NN = reshape(HbO_NN,14,[]);
    HbO_NN = HbO_NN';
    index = (subfolder-1)*n_channels+1 : subfolder*n_channels;
    Real_matrix(index,end) = mean(HbO_NN,1);

    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Real_NN_8layers_act.mat');
    load(filepath)% File exists.
    Hb_NN = Y_real_act;

    HbO_NN = mean(Hb_NN(:, 1:512),2);
    HbO_NN = reshape(HbO_NN,14,[]);
    HbO_NN = HbO_NN';
    Real_matrix_act(index,end) = mean(HbO_NN,1);
end

figure
boxplot(Real_matrix.*1e6, 'positions', (1:size(Real_matrix,2)) - 0.2, 'colors','b', 'widths', 0.3, 'outliersize', 2, 'symbol', 'b.');
hold on
boxplot(Real_matrix_act.*1e6, 'positions', (1:size(Real_matrix,2)) + 0.2, 'colors','r', 'widths', 0.3, 'outliersize', 2, 'symbol', 'r.');
hold on
plot([0 9],[0 0],'k-')
ylabel('Sum \Delta HbO (\muMol)')
set(gca, 'XTick', 1:size(Real_matrix_act,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
xlim([0.5 7.5])
ylim([-15 60])
set(gcf,'Position',[3   431   390   274]);
box off


% sig test
fprintf('For 3C sig test\n')

for i = 1:size(Real_matrix,2)
    x1 = Real_matrix(:,i);
    x2 = Real_matrix_act(:,i);
    h1 = kstest(x1);
    h2 = kstest(x2);
    if h1 == 1
        p1 = signtest(x1, 0, 'tail', 'right');
    else
        [~,p1] = ttest(x1, 0, 'tail', 'right');
    end
    if h2 == 1
        p2 = signtest(x2, 0, 'tail', 'right');
    else
        [~,p2] = ttest(x2, 0, 'tail', 'right');
    end
    
    fprintf('For %s: no act h = %d; p = %.3f; Act h = %d; p = %.3f\n', labels{i}, h1, p1, h2, p2)
end

for i = 1:size(Real_matrix,2)
    x1 = Real_matrix(:,i);
    x2 = Real_matrix_act(:,i);
    x = x2-x1;
    h = kstest(x);
    if h == 1
        p = signtest(x, 0, 'tail', 'right');
    else
        [~,p] = ttest(x, 0, 'tail', 'right');
    end
    
    fprintf('For %s: h = %d; p = %.3f\n', labels{i}, h, p)
end


%% NEW DATASET: The number of residual motion artifacts for the real dataset.

% save No. of MAs for each files n_files X [n_MA_no_correction,dc_avg_PCA97,n_MA_Spline99,n_MA_Wavelet01,]
load('Processed_data/Process_real_data_newdata.mat','Proc_data','MA_matrix')

files = dir(fullfile(DataDir,'Real_NN_8layers_newdata_*.mat'));
n_NN = zeros(size(MA_matrix,1),1);
fs_new = 512/20;
for f = 1:length(files)
    fprintf('file is %s\n', files(f).name)
    filepath = fullfile(DataDir, files(f).name);
    load(filepath)% File exists.
    Hb_NN = Y_real_newdata;
%     filepath = fullfile(DataDir, subfolders(subfolder).name, 'number_array.mat');
%     load(filepath)

    SD1.MeasList = [1,1,1,1;1,1,1,2];
    SD1.MeasListAct = [1 1];
    SD1.Lambda = [760;850];
    SD1.SrcPos = [-2.9017 10.2470 -0.4494];
    SD1.DetPos = [-4.5144 9.0228 -1.6928];
    ppf = [1,1];

    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    n = 0;
    for i = 1:size(HbO_NN,1)
        dc_HbO = HbO_NN(i,:);
        dc_HbR = HbR_NN(i,:);
        dc = [dc_HbO;dc_HbR]';
        dod = hmrConc2OD_modified( dc, SD1, ppf );
%         [~,tIncChAuto] = hmrMotionArtifactByChannel(dod, fs_new, SD1, ones(size(dod,1)), 0.5, 1, 50, 200);
        tIncAuto            =   hmrMotionArtifact(dod,fs_new,SD1,ones(size(dod,1)),0.5,1,50, 200);
        n_MA = count_MA(tIncChAuto);
        n = n + n_MA;
    end
    n_NN(f) = n;
end

MA_matrix = [MA_matrix, n_NN];

figure
errorbar(1:size(MA_matrix,2),mean(MA_matrix,1),std(MA_matrix,[],1), 'color','k','marker','o','MarkerFaceColor','k' )

ylabel('Residual motion artifacts')
labels = {'No correction','Spline','Wavelet','Kalman','PCA','CBSI','DAE'};
set(gca, 'XTick', 1:size(MA_matrix,2),'fontsize',12)
set(gca, 'FontName', 'Arial')
set(gca, 'XTickLabel', labels)
xtickangle(90)
% ylim([0 40])
xlim([0.5 8.5])

set(gcf,'Position',[3   431   390   274]);
box off

fprintf('For 3B sig test\n')
% sig test
for i = 2:size(MA_matrix,2)
    x1 = MA_matrix(:,1);
    x2 = MA_matrix(:,i);
    h = kstest(x1 - x2);
    if h == 1
        p = signtest(x1 - x2);
    else
        p = ttest(x1-x2);
    end
    fprintf('for %s:, h = %d; p = %.3f\n', labels{i}, h, p)
end