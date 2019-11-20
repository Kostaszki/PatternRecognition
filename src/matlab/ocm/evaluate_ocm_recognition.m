close all
clear
clc

load('..\..\..\..\..\MagData\LCV91_OB\mag_burst_08_2015.mat')
load('..\..\..\data\ocm\ocm_currents_340.mat');
load('..\..\..\data\ocm\\ocm_test_dataset_05_08_2015.mat')
I = single(I);

l1 = find(time > addtodate(start, -5, 'minute'));
l2 = find(time > addtodate(stop, 5, 'minute'));
if isempty(l2)
    l2 = length(Bx);
end

l1_c = find(cur_time > addtodate(start, -5, 'minute'));
l2_c = find(cur_time > addtodate(stop, 5, 'minute'));

B(1,:) = Bx(l1(1):l2(1));
B(2,:) = By(l1(1):l2(1));
B(3,:) = Bz(l1(1):l2(1));

mag_time = time(l1(1):l2(1));
mag = single(sqrt(B(1,:).^2 + B(2,:).^2 + B(3,:).^2));

%% Load or define reference frame
Fs = 20;
% Optionally load reference signal
load('..\..\..\data\OCMs\ocm_reference_sig.mat');
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
[distance, distance_times] = match_pattern(ref_sig, mag, I(l1_c(1):l2_c(1)), cur_time(l1_c(1):l2_c(1)), mag_time, threshold);
toc

%%
clearvars ocm_times
tic
ocm_times = select_ocms(distance_times, distance, win_size, threshold);
toc

%%
tic
[TP, FP, FN] = evaluate_ocm(y_times, ocm_times);
toc

recall = TP/(TP + FN); precision = TP/(TP + FP);
F1 = 2*precision*recall/(precision + recall); 

fprintf('Recall: %f, Precision: %f, F1: %f \n', recall, precision, F1);