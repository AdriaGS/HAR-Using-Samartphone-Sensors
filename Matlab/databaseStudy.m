%%%%%%%%%%%%%%%%%%     Author: Adrià Gil Sorribes     %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%            08/02/2018              %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%           Version : 1.0            %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('windowed_interpolated_ddbb.mat', 'ddbb_ss');

activities = {'bikeObj', 'busObj', 'carObj', 'nothingObj', 'trainObj', 'walkObj'};

bikeObj = ddbb_ss('bikeObj');
busObj = ddbb_ss('busObj');
carObj = ddbb_ss('carObj');
nothingObj = ddbb_ss('nothingObj');
trainObj = ddbb_ss('trainObj');
walkObj = ddbb_ss('walkObj');

%% Plotting ----------------------------------------------------------------

%XYZ separate axis for each activity on window w
w = 20;

figure; legend('Walk-acc-x', 'Walk-acc-y', 'Walk-acc-z');

%window1_bike = bikeObj(num2str(length(bikeObj)));
window1_bike = bikeObj(num2str(w));
subplot(2,3,1); hold on;
plot(1:length(window1_bike(:,1)), window1_bike(:,4), 'r')
plot(1:length(window1_bike(:,1)), window1_bike(:,5), 'b')
plot(1:length(window1_bike(:,1)), window1_bike(:,6), 'k')
%legend('Bike-acc-x', 'Bike-acc-y', 'Bike-acc-z')
%ylim([-6 6])
title('BIKE')

%window1_bus = busObj(num2str(length(busObj)));
window1_bus = busObj(num2str(w));
subplot(2,3,2); hold on;
plot(1:length(window1_bus(:,1)), window1_bus(:,4), 'r')
plot(1:length(window1_bus(:,1)), window1_bus(:,5), 'b')
plot(1:length(window1_bus(:,1)), window1_bus(:,6), 'k')
%legend('Bus-acc-x', 'Bus-acc-y', 'Bus-acc-z')
%ylim([-6 6])
title('BUS')

%window1_car = carObj(num2str(length(carObj)));
window1_car = carObj(num2str(w));
subplot(2,3,3); hold on;
plot(1:length(window1_car(:,1)), window1_car(:,4), 'r')
plot(1:length(window1_car(:,1)), window1_car(:,5), 'b')
plot(1:length(window1_car(:,1)), window1_car(:,6), 'k')
%legend('Car-acc-x', 'Car-acc-y', 'Car-acc-z')
%ylim([-6 6])
title('CAR')

%window1_nothing = nothingObj(num2str(length(nothingObj)));
window1_nothing = nothingObj(num2str(w));
subplot(2,3,4); hold on;
plot(1:length(window1_nothing(:,1)), window1_nothing(:,4), 'r')
plot(1:length(window1_nothing(:,1)), window1_nothing(:,5), 'b')
plot(1:length(window1_nothing(:,1)), window1_nothing(:,6), 'k')
%legend('Nothing-acc-x', 'Nothing-acc-y', 'Nothing-acc-z')
%ylim([-6 6])
title('NOTHING')

%window1_train = trainObj(num2str(length(trainObj)));
window1_train = trainObj(num2str(w));
subplot(2,3,5); hold on;
plot(1:length(window1_train(:,1)), window1_train(:,4), 'r')
plot(1:length(window1_train(:,1)), window1_train(:,5), 'b')
plot(1:length(window1_train(:,1)), window1_train(:,6), 'k')
%legend('Train-acc-x', 'Train-acc-y', 'Train-acc-z')
%ylim([-6 6])
title('TRAIN')

%window1_walk = walkObj(num2str(length(walkObj)));
window1_walk = walkObj(num2str(w));
subplot(2,3,6); hold on;
plot(1:length(window1_walk(:,1)), window1_walk(:,4), 'r')
plot(1:length(window1_walk(:,1)), window1_walk(:,5), 'b')
plot(1:length(window1_walk(:,1)), window1_walk(:,6), 'k')
%legend('Walk-acc-x', 'Walk-acc-y', 'Walk-acc-z')
%ylim([-6 6])
title('WALK')

%% Plotting 2

figure;
plot(bike_mean_td(:,7), bike_var_td(:,7), 'rx')
hold on
plot(bus_mean_td(:,7), bus_var_td(:,7), 'bo')
hold on
plot(car_mean_td(:,7), car_var_td(:,7), 'kx')
hold on
plot(nothing_mean_td(:,7), nothing_var_td(:,7), 'g^')
hold on
plot(train_mean_td(:,7), train_var_td(:,7), 'yx')
hold on
plot(walk_mean_td(:,7), walk_var_td(:,7), 'co')
hold on

%% Plotting 3

% general graphics, this will apply to any figure you open
% (groot is the default figure object).
set(groot, ...
'DefaultFigureColor', 'w', ...
'DefaultAxesLineWidth', 0.5, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 8, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultLineLineWidth', 1, ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 8, ...
'DefaultTextFontName', 'Helvetica', ...
'DefaultAxesBox', 'off', ...
'DefaultAxesTickLength', [0.02 0.025]);
 
% set the tickdirs to go out - need this specific order
set(groot, 'DefaultAxesTickDir', 'out');
set(groot, 'DefaultAxesTickDirMode', 'manual');

colors = cbrewer('qual', 'Set1', 10);

firstAct = ddbb_ss('walkObj');
secondAct = ddbb_ss('nothingObj');

var = {'X axis', 'Y axis', 'Z axis'};

data1 = firstAct(num2str(10));
data2 = secondAct(num2str(10));

for i = 4:6

    figure;
    title(['Accelerometer Data for Walking and Nothing in ', char(var(i-3))])
    % rather than a square plot, make it thinner
    violinPlot(std(data1(:, i)), 'histOri', 'left', 'widthDiv', [2 1], 'showMM', 0, ...
        'color',  mat2cell(colors(1, : ), 1));

    violinPlot(std(data2(:, i)), 'histOri', 'right', 'widthDiv', [2 2], 'showMM', 0, ...
        'color',  mat2cell(colors(2, : ), 1));

    set(gca, 'xtick', [0.6 1.4], 'xticklabel', {'Walk', 'Nothing'}, 'xlim', [0.2 1.8]);
    ylabel('Value'); xlabel('Data');

    % add significance stars for each bar
    xticks = get(gca, 'xtick');
    % significance star for the difference
    [~, pval] = ttest(data1(:, 1), data2(:, 1));
    % if mysigstar gets 2 xpos inputs, it will draw a line between them and the
    % sigstars on top
    mysigstar(gca, xticks, 18, pval);
end

%% Plotting 4

figure;
l = 1;

for i = activities
    
    activity = ddbb_ss(char(i));
    subplot(2,3,l); hold on;
    temp_corr_x = zeros(length(activity), 3007);
    temp_corr_y = zeros(length(activity), 3007);
    temp_corr_z = zeros(length(activity), 3007);
    for k = 1:length(activity)

        act = activity(num2str(k));
        temp_corr_x(k,:) = xcorr(act(:,7));
        temp_corr_y(k,:) = xcorr(act(:,8));
        temp_corr_z(k,:) = xcorr(act(:,9));

    end
   
    plot(temp_corr_x')
    %plot(mean(temp_corr_y))
    %plot(mean(temp_corr_z))
    %plot((mean(temp_corr_x)+mean(temp_corr_y)+mean(temp_corr_z)), 'b--')
    title(char(i))
    xlim([0 3007])
    
    l = l + 1;
end