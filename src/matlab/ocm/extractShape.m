function [new_C, aligned_SW] = extractShape(SW, C)
    
    n = size(SW,1);
    m = size(SW,2);
    
    aligned_SW = zeros(n, m);
    
    for i=1:n
        [sw_hat] = metrics.align_sigs(C, SW(i,:));
        aligned_SW(i,:) = sw_hat;
    end
   
    S = aligned_SW' * aligned_SW;
    Q = eye(m, m) - 1/m * ones(m, m);
    M = Q' * S * Q;
    [V,D] = eig(M);
    [E_MAX, E_IDX] = max(max(D));
    new_C(1,:) = V(:,E_IDX)';
    
end

