function [ idx_hat ] = func_modelselector( p_hat, K_max )
%% Construct maximum power support set of size K_max 
[~,idx_sort] = sort( p_hat, 'descend' );

%% Return
idx_hat = sort(idx_sort(1:K_max));


end

