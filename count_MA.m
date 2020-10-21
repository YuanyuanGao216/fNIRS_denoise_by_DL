function n_MA = count_MA(tIncChAuto)
% count number of MA in each channel and then sum them up
n_MA = 0;
for Ch = 1:size(tIncChAuto,2)
    [n,~,~] = CalMotionArtifact(tIncChAuto(:,Ch));
    n_MA = n_MA + n;
end