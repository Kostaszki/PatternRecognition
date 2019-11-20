close all
clear
clc

% %% Load data
load('..\..\..\..\..\MagData\LCV91_OB\mag_burst_08_2015.mat')
load('..\..\..\data\ocm\ocm_currents_340.mat');
load('..\..\..\data\ocm\ocm_threshold.mat');
I = single(I);
Fs = 20; %(Hz)


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

%% Load or define reference signal

% Rectangle as initial reference signal
% ref_time = 0:1/Fs:1.0 - 1/Fs; rect_width = 0.3; rect_dis = 0.5;
% shape = rectpuls(ref_time - rect_dis, rect_width);

% Optionally load reference signal
load('..\..\..\data\OCMs\ocm_reference_sig.mat');
ref_time = 0:1/Fs:(length(shape) - 1)/Fs;

for i=1:size(shape,1)
    ref_sig(2*(i - 1) + 1,:) = single(-1*shape(i,:));
    ref_sig(2*i,:) = single(shape(i,:));
end
win_size = length(ref_sig);
%% Search for pattern (reference pattern) in magnetic field data and current data

thresholds = linspace(0, 1, 5);

TP = zeros(length(thresholds), 1);
FP = zeros(length(thresholds), 1);
FN = zeros(length(thresholds), 1);
tic
for i = 1:length(thresholds)
    [distance, distance_times] = match_pattern(ref_sig, mag, I(l1_c(1):l2_c(1)), cur_time(l1_c(1):l2_c(1)), mag_time, thresholds(i));
    ocm_times = select_ocms(distance_times, distance, win_size, thresholds(i));
    [TP(i), FP(i), FN(i)] = evaluate_ocm(y, ocm_times);
end
toc

recall = TP./(TP + FN); precision = TP./(TP + FP);
F1 = 2*precision.*recall./(precision + recall); 
%% Plot
close all
hFig = figure;
plot(thresholds, F1);
ylabel('F1')
xlabel('threshold')
grid on


hFig = figure;
plot(thresholds, recall);
hold
plot(thresholds, precision, '--');
ylabel('Recall/Precision')
xlabel('threshold')
legend('recall', 'precision')
grid on
