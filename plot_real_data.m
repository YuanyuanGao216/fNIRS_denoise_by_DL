function plot_real_data(t,dc,tInc,Ch,t_span,Resting_HbO,sim_data_HbO)
fs = 10;
if (nargin == 0)
    load('Processed_data/plot_Resting.mat');
end

global fs_new
global rt
global pt
fs_new = 7;
rt =   40; %resting time 5.7s
pt =   512-40; %performance time 65.536s - 5.7s


dc              =   dc * 1e6;
Resting_HbO     =   Resting_HbO * 1e6;
sim_data_HbO    =   sim_data_HbO * 1e6;

figure
%% plot the real data
subplot(311)
hold on
h       =   plot(t,squeeze(dc(:,1,Ch)),'b-','linewidth',1);
yl      =   ylim;
start_p =   strfind(tInc',[1 0]);
end_p   =   strfind(tInc',[0 1]);
if size(start_p) ~=0
    for i = 1:length(start_p)
        p = patch([start_p(i) end_p(i) end_p(i) start_p(i)]./fs_new, ...
            [yl(1) yl(1) yl(2) yl(2)], 'r',...
            'edgecolor','none','FaceAlpha',.3);
    end
end
legend([h p(1)],'\DeltaHbO','Motion artifact', 'location', 'northeastoutside')
title('Experimental data example')
ylabel('\muMol\cdotmm');
% xlabel('Time(s)');
xlim([min(t) max(t)])
set(gca,'fontname','Arial','fontsize',fs)
%% plot the motion artifact
subplot(3,6,7)
hold on
x = t(start_p(2):end_p(2));
plot(x,dc(start_p(2):end_p(2)),'b-','linewidth',1);
title('Motion artifact')
ylabel('\muMol\cdotmm');
% xlabel('Time(s)');
xlim([min(x) max(x)])
set(gca,'fontname','Arial','fontsize',fs)
%% plot the resting data
subplot(3,2,4)
hold on
plot(t_span,Resting_HbO,'b-','linewidth',1);
title('Resting state data')
ylabel('\muMol\cdotmm');
% xlabel('Time(s)');
xlim([min(t_span) max(t_span)])
set(gca,'fontname','Arial','fontsize',fs)
%% plot the simulated data
subplot(3,1,3)
hold on

tp = length(sim_data_HbO);
s = zeros(1,tp);
s((rt):512:tp) = 1;
t = (1:tp)./fs_new;

x = (1:length(sim_data_HbO))./fs_new;
h = plot(x,sim_data_HbO,'b-','linewidth',1);
yl = ylim;
index = find(s == 1);
for i = index
    p = plot([t(i) t(i)],yl,'k:','linewidth',1);
end
legend([h,p],{'Resting fNIRS','Stim'},'Location','northeast')

title('Simulated resting state data')
ylabel('\muMol\cdotmm');
% xlabel('Time(s)');
xlim([min(x) max(x)])
set(gca,'fontname','Arial','fontsize',fs)

set(gcf,'position',[17   356   759   386])
saveas(gcf,'Figures/Resting_data_example.fig')
saveas(gcf,'Figures/Resting_data_example.svg')