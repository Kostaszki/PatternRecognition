close all
clear
clc

%% Load data
load('..\..\..\..\..\MagData\LCV91_OB\mag_burst_08_2015.mat')
load('..\..\..\data\ocm\ocm_currents_340.mat');

I = single(I);
Fs = 20;
% Specify interval of interest
start = datenum(2015, 08, 01, 00, 00 , 00);
stop = datenum(2015, 08, 31, 59, 59, 59);

l1 = find(time > start);
l2 = find(time > stop);
if isempty(l2)
    l2 = length(Bx);
end

l1_c = find(cur_time > start);
l2_c = find(cur_time > stop);

B(1,:) = Bx(l1(1):l2(1));
B(2,:) = By(l1(1):l2(1));
B(3,:) = Bz(l1(1):l2(1));

mag_time = time(l1(1):l2(1));

mag = single(sqrt(B(1,:).^2 + B(2,:).^2 + B(3,:).^2));

%% Load or define reference frame

% Rectangle as initial reference signal
ref_time = 0:1/Fs:1.0 - 1/Fs; rect_width = 0.3; rect_dis = 0.5;
shape = rectpuls(ref_time - rect_dis, rect_width);

for i=1:size(shape,1)
    ref_sig(2*(i - 1) + 1,:) = single(-1*shape(i,:));
    ref_sig(2*i,:) = single(shape(i,:));
end
win_size = length(ref_sig);
%% Search for pattern (reference pattern) in magnetic field data and current data

threshold =  0.3103;
for j = 1:3
    % Identify OCMs in specified time interval
    tic
    [distance, distance_times] = match_pattern(ref_sig, mag, I(l1_c(1):l2_c(1)), cur_time(l1_c(1):l2_c(1)), mag_time, threshold);
    ocm_times = select_ocms(distance_times, distance, win_size, threshold);
    toc

    % Extract magnetic field data for indentified OCM time intervals
    ocm_mag = zeros(length(ocm_times), win_size);
    for i = 1:length(ocm_times)
        l1 = find(mag_time > ocm_times(i, 1));
        l2 = find(mag_time > ocm_times(i, 2));
        if length(mag(l1(1):l2(1))) > win_size
            ocm_mag(i,:) = detrend(mag(l1(1):l2(1)-1));
        else
            ocm_mag(i,:) = detrend(mag(l1(1):l2(1)));
        end
    end

    % Compute new reference shape 
    tic
    [idx, C] = k_shape(ocm_mag, ref_sig, 2);
    toc
    ref_sig = C;
end


figure
plot(ref_time, C)

%save('ocm_reference_sig.mat', 'ref_sig', 'ref_time')
