function [cr, SBD, lag] = mxcorr(s1, s2)
    lags = round(-length(s1)/2):round(length(s1)/2);
    cr = zeros(length(lags), 1);
    for l=1:length(lags)
        if lags(l) < 0
            s1_tmp = s1(1:end-abs(lags(l)));
            s2_tmp = s2(abs(lags(l))+1:end);
        else
            s2_tmp = s2(1:end-lags(l));
            s1_tmp = s1(lags(l)+1:end);
        end
        cr(l) = corr(s1_tmp(:), s2_tmp(:));
    end
    [value, index] = max(abs(cr));
    SBD = 1 - value; 
    lag = lags(index);
end
