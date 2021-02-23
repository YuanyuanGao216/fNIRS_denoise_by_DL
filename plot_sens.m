function plot_sens(var_list,n_list,mse_list,method,var)
if (nargin == 0)
    load(['Processed_data/sens_',method,'.mat']);
end
figure('Renderer', 'painters', 'Position', [1 505 300 400])
subplot(2,1,1)
errorbar(var_list,mean(n_list,2),std(n_list,0,2),'Linewidth',1,'Marker','o','markerfacecolor','b')
xlim([min(var_list),max(var_list)])

ylabel('n')
xlabel(var)
title(method)
set(gca,'FontName','Arial','FontSize',15)

subplot(2,1,2)
errorbar(var_list,mean(mse_list,2),std(mse_list,0,2),'Linewidth',1,'Marker','o','markerfacecolor','b')
xlim([min(var_list),max(var_list)])

ylabel('mse')
xlabel(var)
title(method)
set(gca,'FontName','Arial','FontSize',15)

saveas(gcf,['Figures/sens_',method],'fig')
saveas(gcf,['Figures/sens_',method],'svg')