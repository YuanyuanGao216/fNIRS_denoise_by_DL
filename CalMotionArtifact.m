function [n_MA,runstarts,runends] = CalMotionArtifact(tIncChAuto)

transitions = diff([0; tIncChAuto == 0; 0]);
runstarts = find(transitions == 1);
runends = find(transitions == -1);
if isempty(runstarts)
    n_MA = 0;
else
    n_MA = length(runstarts);
end