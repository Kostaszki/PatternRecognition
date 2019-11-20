close all
clear
clc

load('..\..\..\..\..\MagData\LCV91_OB\mag_burst_08_2015.mat')
load('..\..\..\data\ocm\ocm_currents_340.mat');


I = single(I);
Fs = 20; %(Hz)


% Specify interval of interest
start = datenum(2015,08,01,01,47,20);
stop = datenum(2015,08,01,02,02,00);


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
% ref_time = 0:1/Fs:1.0 - 1/Fs; rect_width = 0.3; rect_dis = 0.5;
% shape = rectpuls(ref_time - rect_dis, rect_width);

% Optionally load reference signal
load('..\..\..\data/OCMs/ocm_reference_sig.mat')
ref_time = 0:1/Fs:(length(shape) - 1)/Fs;

for i=1:size(shape,1)
    ref_sig(2*(i - 1) + 1,:) = single(-1*shape(i,:));
    ref_sig(2*i,:) = single(shape(i,:));
end
win_size = length(ref_sig);
%% Search for pattern (reference pattern) in magnetic field data and current data
threshold = 0.5172;
% return proba: Probabiliy
tic
[similarity, similarity_times] = match_pattern(ref_sig, mag, I(l1_c(1):l2_c(1)), cur_time(l1_c(1):l2_c(1)), mag_time, threshold);
toc

%%
clearvars ocm_times
tic
ocm_times = select_ocms(similarity_times, similarity, win_size, threshold);
toc

%save('ocm_times_test.mat', 'ocm_times');
%% Plot
figure
plot(mag_time, mag)
hold on
for i=1:size(ocm_times,1)
    rectangle('Position', [ocm_times(i,1) 0 (ocm_times(i,2) - ocm_times(i,1)) 100], 'FaceColor', [0 0 1 0.1])
end
hold off
datetick('x','HH:MM:SS','keepticks','keeplimits')
ylabel('B (nT)')
xlabel('Time (HH:MM)')
xlim([start stop])
grid on
