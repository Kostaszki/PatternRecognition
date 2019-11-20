function [aligned_sig] = align_sigs(ref_sig, sig)
% Align signals using xcorr
        
    z_ref_sig = metrics.z_normalise(ref_sig);
    z_sig = metrics.z_normalise(sig);

    [~, ~, shift] = metrics.mxcorr(z_ref_sig, z_sig);
 
    
    if shift >= 0
        aligned_sig(:) = [zeros(1,shift), sig(1:end - shift)];
    else
        aligned_sig(:) = [sig(1 - shift:end), zeros(1, -shift)];
    end

end
