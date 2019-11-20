function [r] = compute_corr(ref_sig, sig)
% Similarity measure that is based on the computation of the
% correlation between two signals
    
    %for-loop over all reference signals
    r = 0; % Initialize with minimum similarity
    for i=1:size(ref_sig,1)
        value = corr(ref_sig(i, :)', sig(:));
        
        if value > r
            r = value;
        end 
        
    end
end

