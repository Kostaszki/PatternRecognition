close all
clear
clc

% Split magnetic field data into intervals of equal length (200 pts). Each can then be classified by the neural network.
% To account for different widths of steepened waves resample the data. For
% more details concerning resampling, normalization and features see
% Ostaszewski et al. Pattern Recognition in Time Series for Space
% Missions:A Rosetta Magnetic Field Case Study or the README

load('..\..\..\..\..\MagData\LCV91_OB\mag_burst_11_2015.mat')

Fs = 20; %(Hz)
fNorm = 0.2/(Fs/2);
[fb,fa] = butter(3, fNorm, 'low');

start = datenum(2015,11,19,00,00,00);
stop = datenum(2015,11,19,23,59,59);

l1 = find(time > start);
l2 = find(time > stop);

if isempty(l2)
    l2 = length(Bx);
end
clearvars B
B(1,:) = Bx(l1(1):l2(1));
B(2,:) = By(l1(1):l2(1));
B(3,:) = Bz(l1(1):l2(1));

mag_time = time(l1(1):l2(1));

% Low pass filter the data to reduce high frequency noise
mag = filtfilt(fb, fa, sqrt(B(1,:).^2 + B(2,:).^2 + B(3,:).^2))';

derivates = (zeros(size(mag,1), size(mag,2)));
derivates2 = (zeros(size(mag,1), size(mag,2)));

derivates(1:end-1,1) = diff(mag);
derivates2(1:end-1,1) = diff(derivates);


win_size = 200; %(pts)

% resampling factors
res_start = 2;
res_stop = 15;
res_step = 2;

%parpool('local',24)

tic
%parfor i=res_start:res_step:res_stop
for i=res_start:res_step:res_stop
    
    res_mag = (resample(mag, 1, i));
    res_mag_diff = (resample(derivates, 1, i));
    res_mag_diff2 = (resample(derivates2, 1, i));
    
    slide = round(10*Fs/i);
    num_win = round((length(res_mag) - win_size)/slide);
    
    sw_probs = zeros(num_win,1);
    sw_time = zeros(num_win,1);
    slidingWindow = 1;
    
    sw_mag = zeros(num_win, win_size, 3);
    zero_entry = zeros(num_win, 1);
    
    for win=1:num_win
       min_mag = min(res_mag(slidingWindow:slidingWindow + win_size-1)); 
       max_mag = max(res_mag(slidingWindow:slidingWindow + win_size-1));
       max_mag_diff = max(abs(res_mag_diff(slidingWindow:slidingWindow + win_size-1)));
       max_mag_diff2 = max(abs(res_mag_diff2(slidingWindow:slidingWindow + win_size-1)));
     
       sw_time(win) = mag_time(slidingWindow*i + round(win_size*i/2));
       if max_mag - min_mag > 5
           sw_mag(win, :, 1) = (res_mag(slidingWindow:slidingWindow + win_size-1) - min_mag)./(max_mag - min_mag);
           sw_mag(win, :, 2) = res_mag_diff(slidingWindow:slidingWindow + win_size-1)./max_mag_diff;
           sw_mag(win, :, 3) = res_mag_diff2(slidingWindow:slidingWindow + win_size-1)./max_mag_diff2;
       else
           zero_entry(win) = 1;
       end
       slidingWindow = slidingWindow + slide;
    end
    
    sw_times{i} = sw_time;
    sw_mags{i} = single(sw_mag);

end
toc
fprintf('Total time \n')
%%
tic
res_facs = res_start:res_step:res_stop;
sw_mag = single(zeros(1,200,3));
sw_time = 0;
for l=1:length(sw_mags)
    sw_mag = [sw_mag; sw_mags{l}];
    sw_time = [sw_time; sw_times{l}];
    sw_mags{l} = [];
    sw_times{l} = [];
end
sw_mag(1,:,:) = [];
sw_time(1) = [];
toc
save('../../data/SW/preprocessed/LSTM_preprocess_test4.mat', 'sw_time', 'sw_mag', 'res_facs')