function [idx, C] = k_shape(SW, C_old, k)
    % Clustering algorithm similar to k-Shape. For more information see
    % Paparrizos et al. k-Shape: Efficient and Accurate Clustering of
    % Time Series (2016)

    C = C_old;
    n = size(SW,1);
    m = size(SW,2);
    
    idx = randi([1 k],1, n);
    figure
    
    idx_hat = 2*ones(n, 1);
    
    iter = 0;
    
    while ~isequal(idx, idx_hat) && iter < 50
        
        for i=1:n
            mindist = Inf;
            for j=1:k
            	r = metrics.compute_corr(C(j,:), SW(i,:));
                dist = 1 - r;
                if dist < mindist
                    mindist = dist;
                    idx(i) = j;
                end
            end
        end
        
        for j=1:k
            SW_hat = zeros(1,m);
            for i=1:n
                if idx(i) == j
                    SW_hat = [SW_hat; SW(i,:)];
                end
            end
            SW_hat(1,:) = [];
            C(j,:) = extractShape(SW_hat, C(j,:));
        end 
        iter = iter + 1;
    end
end

