function plot_sim_data(HRFs_HbO,noise_HbO,noised_HRF_HbO,Resting_HbO)
if (nargin == 0)
    load('Processed_data/Example.mat',...
        'HRFs_HbO','noise_HbO','noised_HRF_HbO','Resting_HbO');
end

global fs_new
global rt
global pt
fs_new = 7;
rt =   40; %resting time 5.7s
pt =   512-40; %performance time 65.536s - 5.7s

HRFs_HbO           =   HRFs_HbO .* 1e6;
noise_HbO          =   noise_HbO .* 1e6;
noised_HRF_HbO     =   noised_HRF_HbO .* 1e6;
Resting_HbO        =   Resting_HbO .* 1e6;

tp = length(HRFs_HbO);
s = zeros(1,tp);
s((rt):512:tp) = 1;
t = (1:tp)./fs_new;
%% plot the simulated HRFs
figure
subplot(411); 
hold on
h = plot(t,HRFs_HbO,'b','linewidth',1);
yl = ylim;
index = find(s == 1);
for i = index
    p = plot([t(i) t(i)],yl,'k:','linewidth',1);
end
xlim([min(t),max(t)])
ylabel('\muMol\cdotmm');
title('Simulated evoked responses')
legend([h,p],{'Evoked response','Stim'},'Location','northeast')
set(gca, 'FontName', 'Arial','fontsize',10)

%% plot noise
subplot(412); 
hold on
h = plot(t,noise_HbO,'b','linewidth',1);
yl = ylim;
index = find(s == 1);
for i = index
    p = plot([t(i) t(i)],yl,'k:','linewidth',1);
end
% ylim([-10 150].*1e-6)
xlim([0 512*5/fs_new])
ylabel('\muMol\cdotmm');
title('Simulated motion artifacts')
legend([h,p],{'Motion artifacts','Stim'},'Location','northeast')
set(gca, 'FontName', 'Arial','fontsize',10)
%%
subplot(413); 
hold on
h = plot(t,Resting_HbO,'b','linewidth',1);
yl = ylim;
index = find(s == 1);
for i = index
    p = plot([t(i) t(i)],yl,'k:','linewidth',1);
end
xlim([0 512*5/fs_new])
ylabel('\muMol\cdotmm');
title('Simulated resting fNIRS data')
legend([h,p],{'Resting fNIRS','Stim'},'Location','northeast')
set(gca, 'FontName', 'Arial','fontsize',10)
%% noised HRF
subplot(414); 
hold on
h = plot(t,noised_HRF_HbO,'b','linewidth',1);
yl = ylim;
index = find(s == 1);
for i = index
    p = plot([t(i) t(i)],yl,'k:','linewidth',1);
end
hold off
xlim([0 512*5/fs_new])
ylabel('\muMol\cdotmm');
xlabel('Second')
title('Simulated noised HRFs')
legend([h,p],{'Noised HRFs','Stim'},'Location','northeast')
set(gca, 'FontName', 'Arial','fontsize',10)

set(gcf,'position',[360   312   759   386])
saveas(gcf,'Figures/Sim_data_example.svg')
saveas(gcf,'Figures/Sim_data_example.fig')