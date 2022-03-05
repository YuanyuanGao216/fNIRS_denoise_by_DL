% now we have multiple testing data set
fprintf('testing -------------------\n')
DataDir = 'Processed_data';
subfolders = dir(DataDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name},'.'));
HbO_mse_no_crct     =   [];
HbO_mse_Spline      =   [];
HbO_mse_Wavelet01   =   [];
HbO_mse_Kalman      =   [];
HbO_mse_Cbsi        =   [];
HbO_mse_NN          =   [];

HbR_mse_no_crct     =   [];
HbR_mse_Spline      =   [];
HbR_mse_Wavelet01   =   [];
HbR_mse_Kalman      =   [];
HbR_mse_Cbsi        =   [];
HbR_mse_NN          =   [];
for subfolder = 1:length(subfolders)
% for subfolder = 1:1
%     fprintf('subfolder is %d\n', subfolder)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Spline.mat');
    load(filepath)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Wavelet01.mat');
    load(filepath)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Kalman.mat');
    load(filepath)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Testing_PCA97.mat');
    load(filepath)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Testing_Cbsi.mat');
    load(filepath)
    filepath = fullfile(DataDir, subfolders(subfolder).name, 'SimulateData.mat');
    load(filepath,'HRF_test_noised','HRF_test')
    
    [m,n] = size(HRF_test_noised);
    HbO_test_noised = HRF_test_noised(1:m/2,:);
    HbO_test = HRF_test(1:m/2,:);
    HbR_test_noised = HRF_test_noised(m/2+1:end,:);
    HbR_test = HRF_test(m/2+1:end,:);
    HbO_no_crct = zeros(size(HbO_Cbsi));
    HbR_no_crct = zeros(size(HbO_Cbsi));
    HbO_real = zeros(size(HbO_Cbsi));
    HbR_real = zeros(size(HbO_Cbsi));

    define_constants

    t  = 1/fs_new:1/fs_new:size(HbO_test_noised,2)/fs_new;
    s  = zeros(1,length(t));
    s((rt):512:length(t)) = 1;
    tIncMan=ones(size(t))';

    for i = 1:size(HbO_test,1)
        dc_HbO  = HbO_test_noised(i,:);
        dc_HbR  = HbR_test_noised(i,:);
        dc      =   [dc_HbO;dc_HbR]';
        [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
        HbO_no_crct(i,:) = dc_avg(:,1)';
        HbR_no_crct(i,:) = dc_avg(:,2)';

        HbO_real(i,:) = HbO_test(i,1:512);
        HbR_real(i,:) = HbR_test(i,1:512);
    end

    filepath = fullfile(DataDir, subfolders(subfolder).name, 'Test_NN_8layers.mat');
    load(filepath)
    HbO_NN = zeros(size(HbO_Cbsi));
    HbR_NN = zeros(size(HbO_Cbsi));
    j = 1;
    for i = 1:5:size(Y_test,1)
        HbO_NN(j,:) = mean(Y_test(i:i+4,1:512),1);
        HbR_NN(j,:) = mean(Y_test(i:i+4,513:end),1);
        j = j + 1;
    end


    HbO_mse_no_crct     =   [HbO_mse_no_crct; mean((HbO_no_crct - HbO_real).^2,2)*1e12];
    HbO_mse_Spline      =   [HbO_mse_Spline; mean((HbO_Spline - HbO_real).^2,2)*1e12];
    HbO_mse_Wavelet01   =   [HbO_mse_Wavelet01; mean((HbO_Wavelet01 - HbO_real).^2,2)*1e12];
    HbO_mse_Kalman      =   [HbO_mse_Kalman; mean((HbO_Kalman - HbO_real).^2,2)*1e12];
    HbO_mse_Cbsi        =   [HbO_mse_Cbsi; mean((HbO_Cbsi - HbO_real).^2,2)*1e12];
    HbO_mse_NN          =   [HbO_mse_NN; mean((HbO_NN - HbO_real).^2,2)*1e12];

    HbR_mse_no_crct     =   [HbR_mse_no_crct; mean((HbR_no_crct - HbR_real).^2,2)*1e12];
    HbR_mse_Spline      =   [HbR_mse_Spline; mean((HbR_Spline - HbR_real).^2,2)*1e12];
    HbR_mse_Wavelet01   =   [HbR_mse_Wavelet01; mean((HbR_Wavelet01 - HbR_real).^2,2)*1e12];
    HbR_mse_Kalman      =   [HbR_mse_Kalman; mean((HbR_Kalman - HbR_real).^2,2)*1e12];
    HbR_mse_Cbsi        =   [HbR_mse_Cbsi; mean((HbR_Cbsi - HbR_real).^2,2)*1e12];
    HbR_mse_NN          =   [HbR_mse_NN; mean((HbR_NN - HbR_real).^2,2)*1e12];

end
labels = {'No correction','Spline','Wavelet01','Kalman', 'Cbsi','DAE'};

fprintf('HbO:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbO_mse_no_crct),std(HbO_mse_no_crct), median(HbO_mse_no_crct), iqr(HbO_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbO_mse_Spline),std(HbO_mse_Spline), median(HbO_mse_Spline), iqr(HbO_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbO_mse_Wavelet01),std(HbO_mse_Wavelet01), median(HbO_mse_Wavelet01), iqr(HbO_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbO_mse_Kalman),std(HbO_mse_Kalman), median(HbO_mse_Kalman), iqr(HbO_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbO_mse_Cbsi),std(HbO_mse_Cbsi), median(HbO_mse_Cbsi), iqr(HbO_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbO_mse_NN),std(HbO_mse_NN), median(HbO_mse_NN), iqr(HbO_mse_NN))

fprintf('HbR:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbR_mse_no_crct),std(HbR_mse_no_crct), median(HbR_mse_no_crct), iqr(HbR_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbR_mse_Spline),std(HbR_mse_Spline), median(HbR_mse_Spline), iqr(HbR_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbR_mse_Wavelet01),std(HbR_mse_Wavelet01), median(HbR_mse_Wavelet01), iqr(HbR_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbR_mse_Kalman),std(HbR_mse_Kalman), median(HbR_mse_Kalman), iqr(HbR_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbR_mse_Cbsi),std(HbR_mse_Cbsi), median(HbR_mse_Cbsi), iqr(HbR_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbR_mse_NN),std(HbR_mse_NN), median(HbR_mse_NN), iqr(HbR_mse_NN))
variables = {'no_crct', 'Spline', 'Wavelet01', 'Kalman','Cbsi'};
% sig test of mse
fprintf('HbO:\n')
for i = 1:5
    x1 = HbO_mse_NN;
    eval(strcat('x2 = HbO_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('h = %d; p = %.3f\n', h, p)
end
fprintf('HbR:\n')
for i = 1:5
    x1 = HbR_mse_NN;
    eval(strcat('x2 = HbR_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('h = %d; p = %.3f\n', h, p)
end
%% real no act
fprintf('real no act---------------------\n')
load('Processed_data/Process_real_data.mat','Proc_data')
% since every subj data has been used as testing data in LOO
HbO_real = zeros(512,14)';
HbR_real = zeros(512,14)';

HbO_mse_no_crct     =   [];
HbO_mse_Spline      =   [];
HbO_mse_Wavelet01   =   [];
HbO_mse_Kalman      =   [];
HbO_mse_PCA97       =   [];
HbO_mse_Cbsi        =   [];
HbO_mse_NN          =   [];

HbR_mse_no_crct     =   [];
HbR_mse_Spline      =   [];
HbR_mse_Wavelet01   =   [];
HbR_mse_Kalman      =   [];
HbR_mse_PCA97       =   [];
HbR_mse_Cbsi        =   [];
HbR_mse_NN          =   [];
for i = 1:length(Proc_data)
% for i = 1:1
    HbO_no_crct     =   squeeze(Proc_data(i).dc_no_crct(1:512,1,:))';
    HbO_Spline      =   squeeze(Proc_data(i).dc_Spline(1:512,1,:))';
    HbO_Wavelet01   =   squeeze(Proc_data(i).dc_Wavelet(1:512,1,:))';
    HbO_Kalman      =   squeeze(Proc_data(i).dc_Kalman(1:512,1,:))';
    HbO_PCA97       =   squeeze(Proc_data(i).dc_PCA97(1:512,1,:))';
    HbO_Cbsi        =   squeeze(Proc_data(i).dc_Cbsi(1:512,1,:))';

    HbR_no_crct     =   squeeze(Proc_data(i).dc_no_crct(1:512,2,:))';
    HbR_Spline      =   squeeze(Proc_data(i).dc_Spline(1:512,2,:))';
    HbR_Wavelet01   =   squeeze(Proc_data(i).dc_Wavelet(1:512,2,:))';
    HbR_Kalman      =   squeeze(Proc_data(i).dc_Kalman(1:512,2,:))';
    HbR_PCA97       =   squeeze(Proc_data(i).dc_PCA97(1:512,2,:))';
    HbR_Cbsi        =   squeeze(Proc_data(i).dc_Cbsi(1:512,2,:))';

    filepath = fullfile(DataDir, subfolders(i).name, 'Real_NN_8layers.mat');
    load(filepath)
    Hb_NN = Y_real;
    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    HbO_NN = reshape(HbO_NN',[],14)';
    HbR_NN = reshape(HbR_NN',[],14)';
    
    dc_HbO = HbO_NN;
    dc_HbR = HbR_NN;
    dc = [dc_HbO;dc_HbR]';
    [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
    Proc_data(i).dc_NN = dc_avg';
    HbO_NN = squeeze(Proc_data(i).dc_NN(1:14,:));
    HbR_NN = squeeze(Proc_data(i).dc_NN(15:end,:));
    for channel = 1:size(HbO_real,1)
        HbO_mse_no_crct     =   [HbO_mse_no_crct; mean((HbO_no_crct(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Spline      =   [HbO_mse_Spline; mean((HbO_Spline(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Wavelet01   =   [HbO_mse_Wavelet01; mean((HbO_Wavelet01(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Kalman      =   [HbO_mse_Kalman; mean((HbO_Kalman(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_PCA97       =   [HbO_mse_PCA97; mean((HbO_PCA97(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Cbsi        =   [HbO_mse_Cbsi; mean((HbO_Cbsi(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_NN          =   [HbO_mse_NN; mean((HbO_NN(channel,:) - HbO_real(channel,:)).^2,2)*1e12];

        HbR_mse_no_crct     =   [HbR_mse_no_crct; mean((HbR_no_crct(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Spline      =   [HbR_mse_Spline; mean((HbR_Spline(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Wavelet01   =   [HbR_mse_Wavelet01; mean((HbR_Wavelet01(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Kalman      =   [HbR_mse_Kalman; mean((HbR_Kalman(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_PCA97       =   [HbR_mse_PCA97; mean((HbR_PCA97(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Cbsi        =   [HbR_mse_Cbsi; mean((HbR_Cbsi(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_NN          =   [HbR_mse_NN; mean((HbR_NN(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
    end
end

labels = {'No correction','Spline','Wavelet','Kalman','PCA797','Cbsi','DAE'};

fprintf('HbO:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbO_mse_no_crct),std(HbO_mse_no_crct), median(HbO_mse_no_crct), iqr(HbO_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbO_mse_Spline),std(HbO_mse_Spline), median(HbO_mse_Spline), iqr(HbO_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbO_mse_Wavelet01),std(HbO_mse_Wavelet01), median(HbO_mse_Wavelet01), iqr(HbO_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbO_mse_Kalman),std(HbO_mse_Kalman), median(HbO_mse_Kalman), iqr(HbO_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbO_mse_PCA97),std(HbO_mse_PCA97), median(HbO_mse_PCA97), iqr(HbO_mse_PCA97))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbO_mse_Cbsi),std(HbO_mse_Cbsi), median(HbO_mse_Cbsi), iqr(HbO_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{7}, mean(HbO_mse_NN),std(HbO_mse_NN), median(HbO_mse_NN), iqr(HbO_mse_NN))

fprintf('HbR:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbR_mse_no_crct),std(HbR_mse_no_crct), median(HbR_mse_no_crct), iqr(HbR_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbR_mse_Spline),std(HbR_mse_Spline), median(HbR_mse_Spline), iqr(HbR_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbR_mse_Wavelet01),std(HbR_mse_Wavelet01), median(HbR_mse_Wavelet01), iqr(HbR_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbR_mse_Kalman),std(HbR_mse_Kalman), median(HbR_mse_Kalman), iqr(HbR_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbR_mse_PCA97),std(HbR_mse_PCA97), median(HbR_mse_PCA97), iqr(HbR_mse_PCA97))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbR_mse_Cbsi),std(HbR_mse_Cbsi), median(HbR_mse_Cbsi), iqr(HbR_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{7}, mean(HbR_mse_NN),std(HbR_mse_NN), median(HbR_mse_NN), iqr(HbR_mse_NN))

% sig test of mse
variables = {'no_crct', 'Spline', 'Wavelet01', 'Kalman', 'PCA97', 'Cbsi'};
fprintf('HbO:\n')
for i = 1:6
    x1 = HbO_mse_NN;
    eval(strcat('x2 = HbO_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
end
fprintf('HbR:\n')
for i = 1:6
    x1 = HbR_mse_NN;
    eval(strcat('x2 = HbR_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
end
%% real act
fprintf('For real act---------------------------\n')
load('Processed_data/Process_real_data.mat','Proc_data','MA_matrix')
HbO_mse_no_crct     =   [];
HbO_mse_Spline      =   [];
HbO_mse_Wavelet01   =   [];
HbO_mse_Kalman      =   [];
HbO_mse_PCA97       =   [];
HbO_mse_Cbsi        =   [];
HbO_mse_NN          =   [];

HbR_mse_no_crct     =   [];
HbR_mse_Spline      =   [];
HbR_mse_Wavelet01   =   [];
HbR_mse_Kalman      =   [];
HbR_mse_PCA97       =   [];
HbR_mse_Cbsi        =   [];
HbR_mse_NN          =   [];
for i = 1:length(Proc_data)
% for i = 7:7
    HbO_real        =   repmat(Proc_data(i).HRF.HbO(1:512),14,1);
    HbO_no_crct     =   squeeze(Proc_data(i).dc_act_no_crct(1:512,1,:))';
    HbO_Spline      =   squeeze(Proc_data(i).dc_act_Spline(1:512,1,:))';
    HbO_Wavelet01   =   squeeze(Proc_data(i).dc_act_Wavelet(1:512,1,:))';
    HbO_Kalman      =   squeeze(Proc_data(i).dc_act_Kalman(1:512,1,:))';
    HbO_PCA97       =   squeeze(Proc_data(i).dc_act_PCA97(1:512,1,:))';
    HbO_Cbsi        =   squeeze(Proc_data(i).dc_act_Cbsi(1:512,1,:))';

    HbR_real        =   repmat(Proc_data(i).HRF.HbR(1:512),14,1);
    HbR_no_crct     =   squeeze(Proc_data(i).dc_act_no_crct(1:512,2,:))';
    HbR_Spline      =   squeeze(Proc_data(i).dc_act_Spline(1:512,2,:))';
    HbR_Wavelet01   =   squeeze(Proc_data(i).dc_act_Wavelet(1:512,2,:))';
    HbR_Kalman      =   squeeze(Proc_data(i).dc_act_Kalman(1:512,2,:))';
    HbR_PCA97       =   squeeze(Proc_data(i).dc_act_PCA97(1:512,2,:))';
    HbR_Cbsi        =   squeeze(Proc_data(i).dc_act_Cbsi(1:512,2,:))';

    filepath = fullfile(DataDir, subfolders(i).name, 'Real_NN_8layers_act.mat');
    load(filepath)
    Hb_NN = Y_real_act;
    HbO_NN = Hb_NN(:, 1:512);
    HbR_NN = Hb_NN(:, 513:end);
    HbO_NN = reshape(HbO_NN',[],14)';
    HbR_NN = reshape(HbR_NN',[],14)';
    
    
    dc_HbO = HbO_NN;
    dc_HbR = HbR_NN;
    dc = [dc_HbO;dc_HbR]';
    [dc_avg, ~, ~, ~, ~, ~] =   hmrBlockAvg(dc, s', t, [-39/fs_new (512-40)/fs_new] ); 
    Proc_data(i).dc_NN = dc_avg';

    HbO_NN = squeeze(Proc_data(i).dc_NN(1:14,:));
    HbR_NN = squeeze(Proc_data(i).dc_NN(15:end,:));
    
    for channel = 1:size(HbO_real,1)
        HbO_mse_no_crct     =   [HbO_mse_no_crct; mean((HbO_no_crct(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Spline      =   [HbO_mse_Spline; mean((HbO_Spline(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Wavelet01   =   [HbO_mse_Wavelet01; mean((HbO_Wavelet01(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Kalman      =   [HbO_mse_Kalman; mean((HbO_Kalman(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_PCA97       =   [HbO_mse_PCA97; mean((HbO_PCA97(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_Cbsi        =   [HbO_mse_Cbsi; mean((HbO_Cbsi(channel,:) - HbO_real(channel,:)).^2,2)*1e12];
        HbO_mse_NN          =   [HbO_mse_NN; mean((HbO_NN(channel,:) - HbO_real(channel,:)).^2,2)*1e12];

        HbR_mse_no_crct     =   [HbR_mse_no_crct; mean((HbR_no_crct(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Spline      =   [HbR_mse_Spline; mean((HbR_Spline(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Wavelet01   =   [HbR_mse_Wavelet01; mean((HbR_Wavelet01(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Kalman      =   [HbR_mse_Kalman; mean((HbR_Kalman(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_PCA97       =   [HbR_mse_PCA97; mean((HbR_PCA97(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_Cbsi        =   [HbR_mse_Cbsi; mean((HbR_Cbsi(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
        HbR_mse_NN          =   [HbR_mse_NN; mean((HbR_NN(channel,:) - HbR_real(channel,:)).^2,2)*1e12];
    end
end

labels = {'No correction','Spline','Wavelet01','Kalman','PCA797','Cbsi','DAE'};

fprintf('HbO:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbO_mse_no_crct),std(HbO_mse_no_crct), median(HbO_mse_no_crct), iqr(HbO_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbO_mse_Spline),std(HbO_mse_Spline), median(HbO_mse_Spline), iqr(HbO_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbO_mse_Wavelet01),std(HbO_mse_Wavelet01), median(HbO_mse_Wavelet01), iqr(HbO_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbO_mse_Kalman),std(HbO_mse_Kalman), median(HbO_mse_Kalman), iqr(HbO_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbO_mse_PCA97),std(HbO_mse_PCA97), median(HbO_mse_PCA97), iqr(HbO_mse_PCA97))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbO_mse_Cbsi),std(HbO_mse_Cbsi), median(HbO_mse_Cbsi), iqr(HbO_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{7}, mean(HbO_mse_NN),std(HbO_mse_NN), median(HbO_mse_NN), iqr(HbO_mse_NN))

fprintf('HbR:\n')
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{1}, mean(HbR_mse_no_crct),std(HbR_mse_no_crct), median(HbR_mse_no_crct), iqr(HbR_mse_no_crct))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{2}, mean(HbR_mse_Spline),std(HbR_mse_Spline), median(HbR_mse_Spline), iqr(HbR_mse_Spline))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{3}, mean(HbR_mse_Wavelet01),std(HbR_mse_Wavelet01), median(HbR_mse_Wavelet01), iqr(HbR_mse_Wavelet01))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{4}, mean(HbR_mse_Kalman),std(HbR_mse_Kalman), median(HbR_mse_Kalman), iqr(HbR_mse_Kalman))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{5}, mean(HbR_mse_PCA97),std(HbR_mse_PCA97), median(HbR_mse_PCA97), iqr(HbR_mse_PCA97))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{6}, mean(HbR_mse_Cbsi),std(HbR_mse_Cbsi), median(HbR_mse_Cbsi), iqr(HbR_mse_Cbsi))
fprintf('for %s: %.2f(%.2f)\t median = %.2f; IQR = %.2f\n', labels{7}, mean(HbR_mse_NN),std(HbR_mse_NN), median(HbR_mse_NN), iqr(HbR_mse_NN))

% sig test of mse
fprintf('HbO:\n')
for i = 1:6
    x1 = HbO_mse_NN;
    eval(strcat('x2 = HbO_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
end
fprintf('HbR:\n')
for i = 1:6
    x1 = HbR_mse_NN;
    eval(strcat('x2 = HbR_mse_',variables{i},';'));
    delta_x = x2 - x1;
    h = kstest(delta_x);
    if h == 1
        p = signtest(delta_x);
    else
        p = ttest(delta_x);
    end
    p2 = ttest(delta_x);
    fprintf('For %s, h = %d; p = %.3f\n',variables{i}, h, p)
end
