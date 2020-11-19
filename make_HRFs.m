function HRF_signal = make_HRFs(s,amp_HbO)

global fs_new

HRF_function = amp_HbO./gamma((1:(512-40))/5/fs_new); % figure; plot(HRF_function);
HRFs = conv(HRF_function,s); % figure; plot(HRFs)
HRF_signal.HbO = HRFs(1:length(s)) * 1e-6;

amp_HbR = - amp_HbO/3;
HRF_function = amp_HbR./gamma((1:(512-40))/5/fs_new); % figure; plot(HRF_function);
HRFs = conv(HRF_function,s); % figure; plot(HRFs)
HRF_signal.HbR = HRFs(1:length(s)) * 1e-6;