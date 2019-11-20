function [TP, FP, FN] = evaluate_ocm(y_times, ocm_times)
    % Check of OCMs were identified
    if isempty(ocm_times)
        TP = 0; FP = 0; FN = length(y_times);
        return
    end
    y_hat = ones(size(ocm_times, 1), 1);
    y = ones(length(y_times), 1);
    true_pos = 0;
    for i = 1:length(y)
        for j = 1:size(ocm_times, 1)
            % An OCM is correctly identified if the given time (y_times(i))
            % lies in the interval identified by the algorithm
            if ocm_times(j, 1) < y_times(i) && y_times(i) < ocm_times(j, 2)
                y_hat(j) = 0;
                true_pos = 1;
                break;
            end
        end
        if y_hat(j) == 0 && true_pos == 1
            y(i) = 0; % TP
        end
        true_pos = 0;
    end
    
    assert(sum(y == 0) == sum(y_hat == 0));
    TP = sum(y == 0); FN = sum(y); FP = sum(y_hat);
    
end

