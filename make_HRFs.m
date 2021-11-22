function HRF_signal = make_HRFs(s,amp_HbO, time_to_peak, duration, sigma)

global fs_new
tHRF = (0:511)/fs_new;
dt = 1./fs_new;

tau = time_to_peak;
T = duration;
% tau = 1;
% sigma = 10;
% T = 10;

tbasis = (exp(1)*(tHRF-tau).^2/sigma^2) .* exp( -(tHRF-tau).^2/sigma^2 );
lstNeg = find(tHRF<0);
tbasis(lstNeg) = 0;
if tHRF(1)<tau
    tbasis(1:round((tau-tHRF(1))/dt)) = 0;
end

tbasis_conv = conv(tbasis,ones(round(T/dt),1)) / round(T/dt);

amp_tbasis = max(tbasis_conv);
HRF_function = tbasis_conv./amp_tbasis .* amp_HbO;


HRFs = conv(HRF_function,s);
HRF_signal.HbO = HRFs(1:length(s)) * 1e-6;

amp_HbR = - amp_HbO/3;
HRF_function = tbasis_conv./amp_tbasis .* amp_HbR;
HRFs = conv(HRF_function,s); 
HRF_signal.HbR = HRFs(1:length(s)) * 1e-6;


