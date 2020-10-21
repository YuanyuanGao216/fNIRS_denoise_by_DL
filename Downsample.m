function fNIRS_data  =   Downsample(fNIRS_data)
global fs_new
t           =   fNIRS_data.t;
fs          =   1/(t(2)-t(1)); % old sampling rate
fsn         =   fs/fs_new;

% revised from hmrNirsFileDownsample
if floor(fsn) == fsn % integer check
    fNIRS_data.d = downsample(fNIRS_data.d,fsn);
    if exist('fNIRS_data.aux')
        if ~isempty(fNIRS_data.aux)
            fNIRS_data.aux = downsample(fNIRS_data.aux,fsn);
        end
    end
    fNIRS_data.t = downsample(fNIRS_data.t,fsn);
    s_sampled = zeros(size(fNIRS_data.t,1),size(fNIRS_data.s,2));
    for j=1:size(fNIRS_data.s,2);
    lst = find(fNIRS_data.s(:,j)==1);
    lst = round(lst/fsn);
    s_sampled(lst,j) = 1;
    end
    for j=1:size(fNIRS_data.s,2);
    lst = find(fNIRS_data.s(:,j)==-1);
    lst = round(lst/fsn);
    s_sampled(lst,j) = -1;
    end
    fNIRS_data.s = s_sampled;
else  % if downsample factor is not an integer (first upsample then downsample)
    t_new = linspace(1, size(fNIRS_data.d,1), 10*size(fNIRS_data.d,1));
    d = interp1(fNIRS_data.d, t_new);
    fNIRS_data.d = downsample(d,round(fsn*10));
    if exist('fNIRS_data.aux')
        if ~isempty(fNIRS_data.aux)
            aux = interp1(fNIRS_data.aux, t_new);
            fNIRS_data.aux = downsample(aux,round(fsn*10))';
        end
    end

    t = interp1(fNIRS_data.t, t_new);
    fNIRS_data.t = downsample(t,round(fsn*10))';

    s_sampled = zeros(size(fNIRS_data.t,1),size(fNIRS_data.s,2));
    for j=1:size(fNIRS_data.s,2);
    lst = find(fNIRS_data.s(:,j)==1);
    lst = round(lst/fsn);
    s_sampled(lst,j) = 1;
    end
    for j=1:size(fNIRS_data.s,2);
    lst = find(fNIRS_data.s(:,j)==-1);
    lst = round(lst/fsn);
    s_sampled(lst,j) = -1;
    end
    fNIRS_data.s = s_sampled;
end