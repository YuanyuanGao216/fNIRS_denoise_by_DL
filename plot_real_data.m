% function plot_real_data(t,dc,tInc,ch,t_span,Resting_HbO,sim_data_HbO)
function plot_real_data(t,dc,tInc,ch)
if (nargin == 0)
    load('Processed_data/plot_Resting.mat');
    ch = channel;
end

global fs_new
fs_new = 7;

dc              =   dc * 1e6;
% Resting_HbO     =   Resting_HbO * 1e6;
% sim_data_HbO    =   sim_data_HbO * 1e6;

figure
%% plot the real data
subplot(311)
hold on
h       =   plot(t,squeeze(dc(:,1,ch)),'b-','linewidth',1);
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
% legend([h p(1)],'\DeltaHbO','Motion artifact')
title('Experimental data example')
ylabel('\muMol');xlabel('Time(s)');xlim([min(t) max(t)])
set(gca,'fontname','Arial','fontsize',10)
% %% plot the motion artifact
% subplot(3,3,4)
% hold on
% x = t(start_p(2):end_p(2));
% plot(x,dc(start_p(2):end_p(2)),'b-','linewidth',1);
% title('Motion artifact')
% ylabel('\muMol');xlabel('Time(s)');xlim([min(x) max(x)])
% set(gca,'fontname','Arial','fontsize',10)
% %% plot the resting data
% subplot(3,2,4)
% hold on
% plot(t_span,Resting_HbO,'b-','linewidth',1);
% title('Resting state data')
% ylabel('\muMol');xlabel('Time(s)');xlim([min(t_span) max(t_span)])
% set(gca,'fontname','Arial','fontsize',10)
% %% plot the simulated data
% subplot(3,1,3)
% hold on
% x = (1:length(sim_data_HbO))./fs_new;
% plot(x,sim_data_HbO,'g-','linewidth',1)
% title('Simulated resting state data')
% ylabel('\muMol');xlabel('Time(s)');xlim([min(x) max(x)])
% set(gca,'fontname','Arial','fontsize',10)
% 
% set(gcf,'position',[17    324    1101    504])
% save('Figures/Resting_data_example.fig')
% save('Figures/Resting_data_example.svg')