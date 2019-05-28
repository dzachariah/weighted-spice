function [ bin_detect, sq_theta_error ] = func_evaluatedetection( theta_hat, theta_true, theta_tol )
%% Initialize
K_true = length(theta_true);
K_hat  = length(theta_hat);

%Initial value
bin_detect = 0;
sq_theta_error = Inf;

%% If model size matches, check estimates
if K_true == K_hat    
    if sum( abs(theta_hat - theta_true) < theta_tol ) == K_true
        
        %Set flag
        bin_detect = 1;
    end
    
    %Compute errors of peaks
    sq_theta_error = norm(theta_hat - theta_true).^2;
        
end



end

