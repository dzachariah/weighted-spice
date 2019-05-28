function [ idx_hat ] = func_peakselector( p_hat, K_max )
%% Find peaks
M = length(p_hat);
p_peak = zeros(size(p_hat));

for k = 2:M-1
    
    %Value greater than left and right
    if (p_hat(k) > p_hat(k+1)) && (p_hat(k) > p_hat(k-1))
        p_peak(k) = 1;
    end
    
end

%% Return
[~,idx_sort] = sort( p_hat .* p_peak, 'descend' );
idx_hat      = sort(idx_sort(1:K_max));

%TEMP:
% close
% plot( p_hat .* p_peak, 'o' ), hold on
% plot( p_hat, '-' )
% pause
 

end

