function [sim_data_HbO,state] = AR_sim(Resting_HbO)

state = 0;
Resting_HbO = Resting_HbO - mean(Resting_HbO);
order = 5;
Md = arima(order,0,0);
try
    EstMd = estimate(Md,Resting_HbO,'Display','off');
catch
    state = 1;
    sim_data_HbO = [];
    return
end

Constant = EstMd.Constant;
AR = cell2mat(EstMd.AR)';
Variance = EstMd.Variance;
EstMd_HbO = arima('Constant',Constant,'AR',AR,'Variance',Variance); 
sim_data_HbO = simulate(EstMd_HbO,512*5,'NumPaths',1);
mean_value = mean(sim_data_HbO);
sim_data_HbO = sim_data_HbO - mean_value;