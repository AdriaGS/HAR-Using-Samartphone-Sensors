%%%%%%%%%%%%%%%%%%     Author: Adrià Gil Sorribes     %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%            10/02/2018              %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%           Version : 1.0            %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%alena = extractfield(dir('Database/alena'), 'name');
%alena = strcat('Database/alena/', alena);
joaquim = extractfield(dir('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/joaquim'), 'name');
joaquim = strcat('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/joaquim/', joaquim);
pau = extractfield(dir('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/pau'), 'name');
pau = strcat('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/pau/', pau);
viet_thi = extractfield(dir('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/viet-thi'), 'name');
viet_thi = strcat('/Users/adriagil/UNI/IPAL/MobilityApps/Activitrack_adria/Database/viet-thi/', viet_thi);

%files = [alena(3:end), joaquim(3:end), pau(3:end), viet_thi(3:end)];
files = [viet_thi(3:end), joaquim(3:end), pau(3:end)];

nothingObj = containers.Map;
nothingC = 1;
walkObj = containers.Map;
walkC = 1;
busObj = containers.Map;
busC = 1;
trainObj = containers.Map;
trainC = 1;
bikeObj = containers.Map;
bikeC = 1;
carObj = containers.Map;
carC = 1;

result = containers.Map;

windowSize = 30;
max_length = 0;

for file = files
    
    display(char(file))
    
    if not(isempty(strfind(char(file), 'Nothing')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                nothingObj(num2str(nothingC)) = result(num2str(i));
                nothingC = nothingC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end
        
    elseif not(isempty(strfind(char(file), 'Walk')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                walkObj(num2str(walkC)) = result(num2str(i));
                walkC = walkC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end
        
    elseif not(isempty(strfind(char(file), 'Bus')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                busObj(num2str(busC)) = result(num2str(i));
                busC = busC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end
        
    elseif not(isempty(strfind(char(file), 'Train')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                trainObj(num2str(trainC)) = result(num2str(i));
                trainC = trainC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end
        
    elseif not(isempty(strfind(char(file), 'Bike')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                bikeObj(num2str(bikeC)) = result(num2str(i));
                bikeC = bikeC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end
 
    elseif not(isempty(strfind(char(file), 'Car')))
        result = csvWindowing(char(file), windowSize);
        sizeRes = size(result);
        for i = 1:sizeRes(1)
            if(length(result(num2str(i))) > 15)
                carObj(num2str(carC)) = result(num2str(i));
                carC = carC + 1;
                if(length(result(num2str(i))) > max_length)
                    max_length = length(result(num2str(i)));
                end
            end
        end   
    end
end

display(max_length)

ddbb = containers.Map;
ddbb('bikeObj') = bikeObj;
ddbb('busObj') = busObj;
ddbb('carObj') = carObj;
ddbb('nothingObj') = nothingObj;
ddbb('trainObj') = trainObj;
ddbb('walkObj') = walkObj;

save('windowed_ddbb.mat', 'ddbb');