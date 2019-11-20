function [zNorm_SW] = z_normalise(SW)
    
    zNorm_SW = zeros(size(SW,1), size(SW,2));
    % z-normalise input signal
    for i=1:size(SW,1)
        sig = SW(i,:);
        mean_sig = mean(sig);
        std_sig = std(sig);
        zNorm_SW(i,:) = (sig - mean_sig)./std_sig;
    end
    
end

