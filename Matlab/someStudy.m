%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%     Author: Adrià Gil Sorribes     %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%            08/02/2018              %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%           Version : 1.0            %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

walkDataset = csvread('Walk.csv');
bikeDataset = csvread('Bike.csv');
trainDataset = csvread('Train.csv');
busDataset = csvread('Bus.csv');
carDataset = csvread('Car.csv');
nothingDataset = csvread('Nothing.csv');

%% Geomagnetic Rotation Vector XYZ

geoRotation_walk_x = walkDataset(:,1);
geoRotation_walk_y = walkDataset(:,2);
geoRotation_walk_z = walkDataset(:,3);
geoRotation_bike_x = bikeDataset(:,1);
geoRotation_bike_y = bikeDataset(:,2);
geoRotation_bike_z = bikeDataset(:,3);
geoRotation_train_x = trainDataset(:,1);
geoRotation_train_y = trainDataset(:,2);
geoRotation_train_z = trainDataset(:,3);
geoRotation_bus_x = busDataset(:,1);
geoRotation_bus_y = busDataset(:,2);
geoRotation_bus_z = busDataset(:,3);
geoRotation_car_x = carDataset(:,1);
geoRotation_car_y = carDataset(:,2);
geoRotation_car_z = carDataset(:,3);
geoRotation_nothing_x = nothingDataset(:,1);
geoRotation_nothing_y = nothingDataset(:,2);
geoRotation_nothing_z = nothingDataset(:,3);

figure;
plot3(geoRotation_walk_x, geoRotation_walk_y, geoRotation_walk_z, 'b*')
hold on
plot3(geoRotation_bike_x, geoRotation_bike_y, geoRotation_bike_z, 'kd')
hold on
plot3(geoRotation_train_x, geoRotation_train_y, geoRotation_train_z, 'c+')
hold on
plot3(geoRotation_bus_x, geoRotation_bus_y, geoRotation_bus_z, 'm.')
hold on
plot3(geoRotation_car_x, geoRotation_car_y, geoRotation_car_z, 'y<')
hold on
plot3(geoRotation_nothing_x, geoRotation_nothing_y, geoRotation_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Geomagnetic Rotation XYZ')

%% Geomagnetic Rotation Vector 2 XY

geoRotation_walk_x2 = walkDataset(:,15);
geoRotation_walk_y2 = walkDataset(:,4);
geoRotation_bike_x2 = bikeDataset(:,15);
geoRotation_bike_y2 = bikeDataset(:,4);
geoRotation_train_x2 = trainDataset(:,15);
geoRotation_train_y2 = trainDataset(:,4);
geoRotation_bus_x2 = busDataset(:,15);
geoRotation_bus_y2 = busDataset(:,4);
geoRotation_car_x2 = carDataset(:,15);
geoRotation_car_y2 = carDataset(:,4);
geoRotation_nothing_x2 = nothingDataset(:,15);
geoRotation_nothing_y2 = nothingDataset(:,4);

figure;
plot(geoRotation_walk_x2, geoRotation_walk_y2, 'b*')
hold on
plot(geoRotation_bike_x2, geoRotation_bike_y2, 'kd')
hold on
plot(geoRotation_train_x2, geoRotation_train_y2, 'c+')
hold on
plot(geoRotation_bus_x2, geoRotation_bus_y2, 'm.')
hold on
plot(geoRotation_car_x2, geoRotation_car_y2, 'y<')
hold on
plot(geoRotation_nothing_x2, geoRotation_nothing_y2, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Geomagnetic Rotation 2 XY')

%% Rotation Vector XYZ

rotation_walk_x = walkDataset(:,9);
rotation_walk_y = walkDataset(:,10);
rotation_walk_z = walkDataset(:,11);
rotation_bike_x = bikeDataset(:,9);
rotation_bike_y = bikeDataset(:,10);
rotation_bike_z = bikeDataset(:,11);
rotation_train_x = trainDataset(:,9);
rotation_train_y = trainDataset(:,10);
rotation_train_z = trainDataset(:,11);
rotation_bus_x = busDataset(:,9);
rotation_bus_y = busDataset(:,10);
rotation_bus_z = busDataset(:,11);
rotation_car_x = carDataset(:,9);
rotation_car_y = carDataset(:,10);
rotation_car_z = carDataset(:,11);
rotation_nothing_x = nothingDataset(:,9);
rotation_nothing_y = nothingDataset(:,10);
rotation_nothing_z = nothingDataset(:,11);

figure;
plot3(rotation_walk_x, rotation_walk_y, rotation_walk_z, 'b*')
hold on
plot3(rotation_bike_x, rotation_bike_y, rotation_bike_z, 'kd')
hold on
plot3(rotation_train_x, rotation_train_y, rotation_train_z, 'c+')
hold on
plot3(rotation_bus_x, rotation_bus_y, rotation_bus_z, 'm.')
hold on
plot3(rotation_car_x, rotation_car_y, rotation_car_z, 'y<')
hold on
plot3(rotation_nothing_x, rotation_nothing_y, rotation_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')

title('Rotation XYZ')

%% Rotation Vector 2 XY

rotation_walk_x2 = walkDataset(:,16);
rotation_walk_y2 = walkDataset(:,5);
rotation_bike_x2 = bikeDataset(:,16);
rotation_bike_y2 = bikeDataset(:,5);
rotation_train_x2 = trainDataset(:,16);
rotation_train_y2 = trainDataset(:,5);
rotation_bus_x2 = busDataset(:,16);
rotation_bus_y2 = busDataset(:,5);
rotation_car_x2 = carDataset(:,16);
rotation_car_y2 = carDataset(:,5);
rotation_nothing_x2 = nothingDataset(:,16);
rotation_nothing_y2 = nothingDataset(:,5);

figure;
plot(rotation_walk_x2, rotation_walk_y2, 'b*')
hold on
plot(rotation_bike_x2, rotation_bike_y2, 'kd')
hold on
plot(rotation_train_x2, rotation_train_y2, 'c+')
hold on
plot(rotation_bus_x2, rotation_bus_y2, 'm.')
hold on
plot(rotation_car_x2, rotation_car_y2, 'y<')
hold on
plot(rotation_nothing_x2, rotation_nothing_y2, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Rotation 2 XY')

%% Accelerometer XYZ

accelerometer_walk_x = walkDataset(:,7);
accelerometer_walk_y = walkDataset(:,6);
accelerometer_walk_z = walkDataset(:,8);
accelerometer_bike_x = bikeDataset(:,7);
accelerometer_bike_y = bikeDataset(:,6);
accelerometer_bike_z = bikeDataset(:,8);
accelerometer_train_x = trainDataset(:,7);
accelerometer_train_y = trainDataset(:,6);
accelerometer_train_z = trainDataset(:,8);
accelerometer_bus_x = busDataset(:,7);
accelerometer_bus_y = busDataset(:,6);
accelerometer_bus_z = busDataset(:,8);
accelerometer_car_x = carDataset(:,7);
accelerometer_car_y = carDataset(:,6);
accelerometer_car_z = carDataset(:,8);
accelerometer_nothing_x = nothingDataset(:,7);
accelerometer_nothing_y = nothingDataset(:,6);
accelerometer_nothing_z = nothingDataset(:,8);

figure;
plot3(accelerometer_walk_x, accelerometer_walk_y, accelerometer_walk_z, 'b*')
hold on
plot3(accelerometer_bike_x, accelerometer_bike_y, accelerometer_bike_z, 'kd')
hold on
plot3(accelerometer_train_x, accelerometer_train_y, accelerometer_train_z, 'c+')
hold on
plot3(accelerometer_bus_x, accelerometer_bus_y, accelerometer_bus_z, 'm.')
hold on
plot3(accelerometer_car_x, accelerometer_car_y, accelerometer_car_z, 'y<')
hold on
plot3(accelerometer_nothing_x, accelerometer_nothing_y, accelerometer_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Accelerometer XYZ')

%% Gravity XYZ

gravity_walk_x = walkDataset(:,13);
gravity_walk_y = walkDataset(:,14);
gravity_walk_z = walkDataset(:,12);
gravity_bike_x = bikeDataset(:,13);
gravity_bike_y = bikeDataset(:,14);
gravity_bike_z = bikeDataset(:,12);
gravity_train_x = trainDataset(:,13);
gravity_train_y = trainDataset(:,14);
gravity_train_z = trainDataset(:,12);
gravity_bus_x = busDataset(:,13);
gravity_bus_y = busDataset(:,14);
gravity_bus_z = busDataset(:,12);
gravity_car_x = carDataset(:,13);
gravity_car_y = carDataset(:,14);
gravity_car_z = carDataset(:,12);
gravity_nothing_x = nothingDataset(:,13);
gravity_nothing_y = nothingDataset(:,14);
gravity_nothing_z = nothingDataset(:,12);

figure;
plot3(gravity_walk_x, gravity_walk_y, gravity_walk_z, 'b*')
hold on
plot3(gravity_bike_x, gravity_bike_y, gravity_bike_z, 'kd')
hold on
plot3(gravity_train_x, gravity_train_y, gravity_train_z, 'c+')
hold on
plot3(gravity_bus_x, gravity_bus_y, gravity_bus_z, 'm.')
hold on
plot3(gravity_car_x, gravity_car_y, gravity_car_z, 'y<')
hold on
plot3(gravity_nothing_x, gravity_nothing_y, gravity_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Gravity XYZ')

%% Linear Acceleration XYZ 

linacceleration_walk_x = walkDataset(:,18);
linacceleration_walk_y = walkDataset(:,19);
linacceleration_walk_z = walkDataset(:,17);
linacceleration_bike_x = bikeDataset(:,18);
linacceleration_bike_y = bikeDataset(:,19);
linacceleration_bike_z = bikeDataset(:,17);
linacceleration_train_x = trainDataset(:,18);
linacceleration_train_y = trainDataset(:,19);
linacceleration_train_z = trainDataset(:,17);
linacceleration_bus_x = busDataset(:,18);
linacceleration_bus_y = busDataset(:,19);
linacceleration_bus_z = busDataset(:,17);
linacceleration_car_x = carDataset(:,18);
linacceleration_car_y = carDataset(:,19);
linacceleration_car_z = carDataset(:,17);
linacceleration_nothing_x = nothingDataset(:,18);
linacceleration_nothing_y = nothingDataset(:,19);
linacceleration_nothing_z = nothingDataset(:,17);

figure;
plot3(linacceleration_walk_x, linacceleration_walk_y, linacceleration_walk_z, 'b*')
hold on
plot3(linacceleration_bike_x, linacceleration_bike_y, linacceleration_bike_z, 'kd')
hold on
plot3(linacceleration_train_x, linacceleration_train_y, linacceleration_train_z, 'c+')
hold on
plot3(linacceleration_bus_x, linacceleration_bus_y, linacceleration_bus_z, 'm.')
hold on
plot3(linacceleration_car_x, linacceleration_car_y, linacceleration_car_z, 'y<')
hold on
plot3(linacceleration_nothing_x, linacceleration_nothing_y, linacceleration_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Linear Acceleration XYZ')

%% Magnetic Field XYZ

magneticField_walk_x = walkDataset(:,21);
magneticField_walk_y = walkDataset(:,22);
magneticField_walk_z = walkDataset(:,20);
magneticField_bike_x = bikeDataset(:,21);
magneticField_bike_y = bikeDataset(:,22);
magneticField_bike_z = bikeDataset(:,20);
magneticField_train_x = trainDataset(:,21);
magneticField_train_y = trainDataset(:,22);
magneticField_train_z = trainDataset(:,20);
magneticField_bus_x = busDataset(:,21);
magneticField_bus_y = busDataset(:,22);
magneticField_bus_z = busDataset(:,20);
magneticField_car_x = carDataset(:,21);
magneticField_car_y = carDataset(:,22);
magneticField_car_z = carDataset(:,20);
magneticField_nothing_x = nothingDataset(:,21);
magneticField_nothing_y = nothingDataset(:,22);
magneticField_nothing_z = nothingDataset(:,20);

figure;
plot3(magneticField_walk_x, magneticField_walk_y, magneticField_walk_z, 'b*')
hold on
plot3(magneticField_bike_x, magneticField_bike_y, magneticField_bike_z, 'kd')
hold on
plot3(magneticField_train_x, magneticField_train_y, magneticField_train_z, 'c+')
hold on
plot3(magneticField_bus_x, magneticField_bus_y, magneticField_bus_z, 'm.')
hold on
plot3(magneticField_car_x, magneticField_car_y, magneticField_car_z, 'y<')
hold on
plot3(magneticField_nothing_x, magneticField_nothing_y, magneticField_nothing_z, 'gp')
legend('walk', 'bike', 'train', 'bus', 'car', 'nothing')
title('Magnetic Field XYZ')

%% K-NN 

%Geomagnetic Rotation
Xtrain_geoRot = [geoRotation_walk_x(2:end), geoRotation_walk_y(2:end), geoRotation_walk_z(2:end); ...
    geoRotation_bike_x(2:end), geoRotation_bike_y(2:end), geoRotation_bike_z(2:end); ...
    geoRotation_train_x(2:end), geoRotation_train_y(2:end), geoRotation_train_z(2:end); ...
    geoRotation_bus_x(2:end), geoRotation_bus_y(2:end), geoRotation_bus_z(2:end); ...
    geoRotation_car_x(2:end), geoRotation_car_y(2:end), geoRotation_car_z(2:end); ...
    geoRotation_nothing_x(2:end), geoRotation_nothing_y(2:end), geoRotation_nothing_z(2:end)];
Ytrain = [ones(1, length(geoRotation_walk_x(2:end))), ones(1, length(geoRotation_bike_x(2:end)))*2, ones(1, length(geoRotation_train_x(2:end)))*3, ...
    ones(1, length(geoRotation_bus_x(2:end)))*4, ones(1, length(geoRotation_car_x(2:end)))*5, ones(1, length(geoRotation_nothing_x(2:end)))*6]';

Mdl = fitcknn(Xtrain_geoRot,Ytrain,'NumNeighbors',3,'Standardize',1);

Xtest_walk = [geoRotation_walk_x(1), geoRotation_walk_y(1), geoRotation_walk_z(1)];
Xtest_bike = [geoRotation_bike_x(1), geoRotation_bike_y(1), geoRotation_bike_z(1)];
Xtest_train = [geoRotation_train_x(1), geoRotation_train_y(1), geoRotation_train_z(1)];
Xtest_bus = [geoRotation_bus_x(1), geoRotation_bus_y(1), geoRotation_bus_z(1)];
Xtest_car = [geoRotation_car_x(1), geoRotation_car_y(1), geoRotation_car_z(1)];
Xtest_nothing = [geoRotation_nothing_x(1), geoRotation_nothing_y(1), geoRotation_nothing_z(1)];
    
[label,score,cost] = predict(Mdl,Xtest_nothing);

display(label)
display(score)

%% K-NN


