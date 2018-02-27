%%%%%%%%%%%%%%%%%%     Author: Adrià Gil Sorribes     %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%            10/02/2018              %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%           Version : 1.0            %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('windowed_ddbb.mat', 'ddbb');

activities = {'bikeObj', 'busObj', 'carObj', 'nothingObj', 'trainObj', 'walkObj'};
ddbb_ss = containers.Map;

max_length = 1504;

for k = activities
    activity = ddbb(char(k));
    temp_map = containers.Map;
    for i = 1:length(activity)
        window = activity(num2str(i));
        windowSize = size(window);
        temp = zeros(max_length, windowSize(2));
        for j = 1:windowSize(2)
            
            temp(:,j) = interp1(1:windowSize(1), window(:,j), linspace(1,windowSize(1),max_length))';

        end
        temp_map(num2str(i)) = temp;
    end
    ddbb_ss(char(k)) = temp_map;
end

save('windowed_interpolated_ddbb.mat', 'ddbb_ss');