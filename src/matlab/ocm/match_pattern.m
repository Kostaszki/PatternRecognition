function [similarity, win_times, num_int] = match_pattern(ref_sig, mag, I, cur_time,  time, threshold)
    
    Fs = 20; %(Hz)
    % Window size corresponds to length of reference signal
    win_size = length(ref_sig); %(pts)
    
    % Define sliding window increment
    slide_increment = 1; %(pts)
    
    % Calculate number of windows
    num_int = round((length(mag) - win_size)/slide_increment);
    % Calculate beginnings of windows
    start_ints(:) = ((1:num_int) - 1)*slide_increment + 1;
  
    % Initialize arrays for computed distance between signal in current
    % window and reference signal
    similarity = zeros(num_int, 1);
    
    % Store physical times for in the middle of window (UTC)
    win_times = zeros(num_int, 1);
    for j=1:length(start_ints)
        win_times(j) = time(round(start_ints(j) + win_size/2));
    end
    
    % Search for switch on signal in houskeeping current
    I_idx = findchangepts(I, 'MinThreshold', 0.1, 'Statistic', 'mean');

    % If no switch on process is found, return sims and win_times
    % initialized with zeros
    if isempty(I_idx)    
        return
    end
    
    % Look for pattern in current 
    idx = 1;
    I_ints = cell(length(I_idx), 1);
    for l=1:length(I_idx) - 1
        I_ints{idx} = find(time(start_ints) > addtodate(cur_time(I_idx(l)), -2, 'minute') & ...
                           time(start_ints) < addtodate(cur_time(I_idx(l+1)), 2, 'minute') & ...
                           mean(I(I_idx(l):I_idx(l + 1))) < 0 & ...
                           (cur_time(I_idx(l+1)) - cur_time(I_idx(l)))*24*60 < 20);
        if ~isempty(I_ints{idx})
            idx = idx + 1;
        end              
    end
    
    I_ints = I_ints(~cellfun('isempty',I_ints));

    
    for j=1:length(I_ints)
        int_idx = I_ints{j};
        
        %parfor i=int_idx(1):int_idx(end)
        for i = int_idx(1):int_idx(end)
            % Start and end of current window
            int_start = start_ints(i);
            int_stop = int_start + win_size-1;

            % Detrend signal in current window
            opol = 2;
            dt_int_B = detrend(mag(int_start:int_stop));
            mag_time = 0:1/Fs:(length(dt_int_B) - 1)/Fs;
            [p, ~, mu] = polyfit(mag_time, dt_int_B, opol);
            f_y = polyval(p, mag_time ,[],mu);
            int_B = dt_int_B - f_y;
            
            % Compute similarity between reference signal and signal in current window.
            r = metrics.compute_corr(ref_sig, int_B);
            
            if r > threshold
                %Check if prominence of identified OCM is at least a value
                %of 1
                [~, locs_p] = findpeaks(int_B, 'MinPeakDistance', length(int_B) - 2, 'MinPeakProminence', 1);
                [~, locs_m] = findpeaks(-1*int_B, 'MinPeakDistance', length(int_B) - 2, 'MinPeakProminence', 1);
                if isempty(locs_p) && isempty(locs_m)
                    r = 0.0;
                end
            end
            similarity(i) = r;
        end
    end
end

