function [ocm_times] = select_ocms(distance_time, distance, win_size, threshold)

    Fs = 20; %(Hz)
    [~, locs] = findpeaks(distance, 'MinPeakHeight', threshold, 'MinPeakDistance', win_size*3/4);
    ocm_times = zeros(length(locs), 2);
    for j=1:length(locs)
        %peak_time = find(time > distance_time(locs(j)));
        ocm_times(j, 1) = addtodate(distance_time(locs(j)), - win_size/2/Fs*1e3, 'millisecond');
        ocm_times(j, 2) = addtodate(distance_time(locs(j)), + win_size/2/Fs*1e3, 'millisecond');
     end
end

