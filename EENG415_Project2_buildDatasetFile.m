%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clayton Daly, Jesse Dugan, Holly Hammons, Luke Logan
% EENG 415
% Dr. Salman Mohagheghi 
% 5/6/2022
% Project 2: Data Processing Code
% NOTE: you will need the file "rsfmeasureddata2011.csv"
% in the same directory to run this file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all

data_table = readtable('rsfmeasureddata2011.csv');
building_net = data_table(:,11);
building_net = table2array(building_net);
building_net(isnan(building_net)) = 0;

A = reshape(building_net, 24, 365);

peak_demand = zeros(1,365);
peak_demand_time = zeros(1,365);
min_demand = zeros(1,365);
min_demand_time = zeros(1,365);
variance = zeros(1,365);
total_energy = zeros(1,365);
load_factor = zeros(1,365);
roc = zeros(23,365);
indx = zeros(1,365);
max_roc = zeros(1,365);

for i = 1:365
    peak_demand(i) = max(A(:,i));
    [~, peak_demand_time(i)] = max(A(:,i));
    min_demand(i) = min(A(:,i));
    [~, min_demand_time(i)] = min(A(:,i));
    variance(i) = var(A(:,i));
    total_energy(i) = trapz(A(:,i));
    load_factor(i) = total_energy(i) / (peak_demand(i) * 24);
    for j = 1:23
        roc(j,i) = A(j+1,i) - A(j,i);
        [~, indx(i)] = max(abs(roc(:,i)));
        max_roc(i) = roc(indx(i), i);
    end
end

hour_bfr = indx;

dataset = [peak_demand' peak_demand_time' min_demand' min_demand_time' max_roc' hour_bfr' ...
    variance' total_energy' load_factor'];

DATASET = array2table(dataset, 'VariableNames', {'PEAK','PEAK TIME', 'MIN', 'MIN TIME'...
    'MAXROC', 'HOURBEFORE', 'VAR', 'TOTAL', 'LOADFAC'});

writetable(DATASET, 'loadprofiles.csv');
