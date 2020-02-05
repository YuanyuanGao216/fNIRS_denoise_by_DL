function sim_data = AR_simulate(Resting_HbO,m)

order = 5;
Md = arima(order,0,0);
try
    EstMd = estimate(Md,Resting_HbO,'Display','off');
catch
    continue
end
Constant = EstMd.Constant;
AR = cell2mat(EstMd.AR)';
Variance = EstMd.Variance;
EstMd_HbO = arima('Constant',Constant,'AR',AR,'Variance',Variance); 
sim_data = simulate(EstMd_HbO,fs*pt,'NumPaths',m);