clear all, clc

%% Seed
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',1));
RandStream.setGlobalStream(RandStream('mt19937ar','seed','shuffle'));


%% Set parameters
%MC parameters
MC_runs  = 1000;
N_dim_list = round(linspace(10,100,8));

%Array and grid dimensions
%N_dim     = 35
M_grid    = 1000
theta_set = linspace(-pi/2,pi/2,M_grid);

%Signal parameters
idx_true    = [400 420 600] %true support set
P_diag_true = [1 3 2].^2            %signal power distribution
delta       = 1/2;                  %wavelength separation
SNR         = 20; %[dB]

%% Resulting parameters
%Unknown structures
P_true     = diag(P_diag_true);
K_true     = length(P_diag_true);
theta_true = theta_set(idx_true); % DOA


%Fixed structures
%I_N       = eye( max(N_dim_list) );
%A_steer_max   = func_ULA( theta_set, delta, max(N_dim_list) );
%A_aug_max     = [A_steer, I_N];


%% Algorithm parameters
%Common
conv_tol      = 1e-3; %tol |p^{i+1} - p^{i}|/|p^{i}| at each step
N_powerupdate = 1;
K_max         = K_true; %TODO
theta_tol     = abs(theta_set(idx_true(1)) - theta_set(idx_true(2)))/2;%1/N_dim;%25 * (theta_set(2) - theta_set(1));


%% Allocate memory to store
%Variable length
N_var = length(N_dim_list);

%Squared errors (oracle)
sq_error_oracle = zeros(1,N_var,MC_runs);

%Squared errors
sq_error_spice = zeros(1,N_var,MC_runs);
sq_error_likes = zeros(1,N_var,MC_runs);
sq_error_slim  = zeros(1,N_var,MC_runs);
sq_error_iaa   = zeros(1,N_var,MC_runs);

%Squared errors (postprocessed estimates)
sq_error_spice_lmvue = zeros(1,N_var,MC_runs);
sq_error_likes_lmvue = zeros(1,N_var,MC_runs);
sq_error_slim_lmvue  = zeros(1,N_var,MC_runs);
sq_error_iaa_lmvue   = zeros(1,N_var,MC_runs);

%Support set
bin_supp_spice = zeros(1,N_var,MC_runs);
bin_supp_likes = zeros(1,N_var,MC_runs);
bin_supp_slim  = zeros(1,N_var,MC_runs);
bin_supp_iaa   = zeros(1,N_var,MC_runs);

%Iteration count
iter_spice = zeros(1,N_var,MC_runs);
iter_likes = zeros(1,N_var,MC_runs);
iter_slim  = zeros(1,N_var,MC_runs);
iter_iaa   = zeros(1,N_var,MC_runs);

%Clock
clock_spice = zeros(1,N_var,MC_runs);
clock_likes = zeros(1,N_var,MC_runs);
clock_slim  = zeros(1,N_var,MC_runs);
clock_iaa   = zeros(1,N_var,MC_runs);

%Theta error
sq_theta_error_spice = zeros(1,N_var,MC_runs);
sq_theta_error_likes = zeros(1,N_var,MC_runs);
sq_theta_error_slim  = zeros(1,N_var,MC_runs);
sq_theta_error_iaa   = zeros(1,N_var,MC_runs);


%Additional:
bin_supp_beam       = zeros(1,N_var,MC_runs);
sq_theta_error_beam = zeros(1,N_var,MC_runs);

%% Monte Carlo
T_snap = 1;
N_dim_max = max(N_dim_list);

clockstart = clock;
for m = 1:MC_runs
    
    %DISP:
    disp('MC:')
    disp('------------------')
    disp(m)
    
    %Generate SOI and regressor
    %---------------------------
    %Generate SOI (complex Gaussian)
    x_true_sub         = chol(P_true)' * exp(1i*2*pi*rand(K_true, T_snap)); %random phase
    x_true             = zeros(M_grid,1);
    x_true(idx_true,1) = x_true_sub;
    
    %Generate max steer
    A_steer_max = func_ULA( theta_set, delta, N_dim_max );
    
    %Generate SOI and regressor
    %---------------------------
    sigma2     = (trace(P_true)) /( 1 * 10^(SNR/10) ); %NOTE: definition
        
    %Generate noise
    n_true_max = sqrt(sigma2) * (randn(N_dim_max,T_snap) + 1i*randn(N_dim_max,T_snap))/sqrt(2);
    
    
    %Vary SNR
    %---------------------------
    count = 1;
    for N_dim = N_dim_list
        
        %DISP:
        disp('N:')
        disp(N_dim)
        
        %Generate regressor
        A_steer    = A_steer_max(1:N_dim,:);
        A_aug     = [A_steer, eye(N_dim)];
        
        %Generate observation
        %---------------------------
        %Noise vector
        n_true = n_true_max(1:N_dim);
        
        %Generate array output
        y = (A_steer*x_true) + n_true;
        
        
        %Compute oracle and beamformer
        %---------------------------
        x_hat_oracle             = zeros(M_grid,1);
        x_hat_oracle(idx_true,1) = A_steer(:,idx_true) \ y;
        
        p_tmp = abs( A_steer'*y ).^2;        
        idx_hat_beam = func_peakselector( real(p_tmp(1:M_grid)), K_max );

        
        %Compute parameter estimates
        %---------------------------
        %SPICE
        disp('SPICE-1')
        type_algorithm = 1; flag_version     = 0; N_powerupdate = 1; iter_limit = 1e3;
        [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );
        
        idx_hat_spice              = func_peakselector( real(p_tmp(1:M_grid)), K_max );
        x_hat_spice                = func_lmmse( y, A_aug, p_tmp );
        x_hat_spice_lmvue          = func_lmvue( y, A_aug, p_tmp );
        iter_spice(1,count,m)      = iter_tmp;
        clock_spice(1,count,m)     = clock_tmp;
        p_hat_spice                = p_tmp; %for likes
        
        %LIKES
        disp('SPICE-2')
        type_algorithm = 2; flag_version     = 0; N_powerupdate = 30; iter_limit = 1e3;
        [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, p_hat_spice );

        idx_hat_likes              = func_peakselector( real(p_tmp(1:M_grid)), K_max );
        x_hat_likes                = func_lmmse( y, A_aug, p_tmp );
        x_hat_likes_lmvue          = func_lmvue( y, A_aug, p_tmp );
        iter_likes(1,count,m)      = iter_tmp;
        clock_likes(1,count,m)     = clock_tmp + clock_spice(1,count,m); %total time
        
        
        %SLIM
        disp('SPICE-3')
        type_algorithm = 3; flag_version     = 0; N_powerupdate = 1; iter_limit = 5;
        [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );

        idx_hat_slim               = func_peakselector( real(p_tmp(1:M_grid)), K_max );
        x_hat_slim                 = func_lmmse( y, A_aug, p_tmp );
        x_hat_slim_lmvue           = func_lmvue( y, A_aug, p_tmp );
        iter_slim(1,count,m)       = iter_tmp;
        clock_slim(1,count,m)      = clock_tmp;

        
        %IAA
        disp('SPICE-4')
        type_algorithm = 4; flag_version     = 0; N_powerupdate = 1; iter_limit = 1e3;
        [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );
        
        idx_hat_iaa                = func_peakselector( real(p_tmp(1:M_grid)), K_max );
        x_hat_iaa                  = func_lmmse( y, A_aug, p_tmp );
        x_hat_iaa_lmvue            = func_lmvue( y, A_aug, p_tmp );
        iter_iaa(1,count,m)        = iter_tmp;
        clock_iaa(1,count,m)       = clock_tmp;
        
        

        
        %Compute squared errors
        %---------------------------
        %Oracle
        sq_error_oracle(1,count,m) = norm( x_true - x_hat_oracle )^2;
        
        %SPICE estimates
        sq_error_spice(1,count,m) = norm( x_true - x_hat_spice )^2;
        sq_error_likes(1,count,m) = norm( x_true - x_hat_likes )^2;
        sq_error_slim(1,count,m)  = norm( x_true - x_hat_slim )^2;
        sq_error_iaa(1,count,m)   = norm( x_true - x_hat_iaa )^2;

        %Postprocessed estimates
        sq_error_spice_lmvue(1,count,m) = norm( x_true - x_hat_spice_lmvue )^2;
        sq_error_likes_lmvue(1,count,m) = norm( x_true - x_hat_likes_lmvue )^2;
        sq_error_slim_lmvue(1,count,m)  = norm( x_true - x_hat_slim_lmvue )^2;
        sq_error_iaa_lmvue(1,count,m)   = norm( x_true - x_hat_iaa_lmvue )^2;

        
        %Compute support set detection & DOA error
        %-------------------------------------------
        [bin_supp_spice(1,count,m), sq_theta_error_spice(1,count,m)] = func_evaluatedetection( theta_set(idx_hat_spice), theta_set(idx_true), theta_tol );
        [bin_supp_likes(1,count,m), sq_theta_error_likes(1,count,m)] = func_evaluatedetection( theta_set(idx_hat_likes), theta_set(idx_true), theta_tol );
        [bin_supp_slim(1,count,m),  sq_theta_error_slim(1,count,m)]  = func_evaluatedetection( theta_set(idx_hat_slim), theta_set(idx_true), theta_tol );
        [bin_supp_iaa(1,count,m),   sq_theta_error_iaa(1,count,m)]   = func_evaluatedetection( theta_set(idx_hat_iaa), theta_set(idx_true), theta_tol );
        
        [bin_supp_beam(1,count,m),   sq_theta_error_beam(1,count,m)]   = func_evaluatedetection( theta_set(idx_hat_beam), theta_set(idx_true), theta_tol );
        

        %Update counter
        %-----------------------------
        count = count + 1;
        
    end
    
    
end
disp('Elapsed time:')
disp(etime(clock, clockstart))


%% Save data
save MC_ULA_var_N

%% Compute statistics

%Oracle
MSE_oracle = mean(sq_error_oracle,3);

%SPICE estimates
NMSE_spice = mean(sq_error_spice,3) / sum(P_diag_true);%./ MSE_oracle;
NMSE_likes = mean(sq_error_likes,3) / sum(P_diag_true);%./ MSE_oracle;
NMSE_slim  = mean(sq_error_slim,3)  / sum(P_diag_true);%./ MSE_oracle;
NMSE_iaa   = mean(sq_error_iaa,3)   / sum(P_diag_true);%./ MSE_oracle;

NMSE_oracle = mean(sq_error_oracle,3) / sum(P_diag_true);

%Postprocessed estimates
NMSE_spice_lmvue = mean(sq_error_spice_lmvue,3) / sum(P_diag_true);
NMSE_likes_lmvue = mean(sq_error_likes_lmvue,3) / sum(P_diag_true);
NMSE_slim_lmvue  = mean(sq_error_slim_lmvue,3)  / sum(P_diag_true);
NMSE_iaa_lmvue   = mean(sq_error_iaa_lmvue,3)   / sum(P_diag_true);

%Probability of detection
PrD_spice = mean(bin_supp_spice,3);
PrD_likes = mean(bin_supp_likes,3);
PrD_slim  = mean(bin_supp_slim,3);
PrD_iaa   = mean(bin_supp_iaa,3);

PrD_beam   = mean(bin_supp_beam,3);


%Iterations
mean_iter_spice = mean(iter_spice,3);
mean_iter_likes = mean(iter_likes,3);
mean_iter_slim  = mean(iter_slim,3);
mean_iter_iaa   = mean(iter_iaa,3);

%Clock
mean_clock_spice = mean(clock_spice,3);
mean_clock_likes = mean(clock_likes,3);
mean_clock_slim  = mean(clock_slim,3);
mean_clock_iaa   = mean(clock_iaa,3);

%DOA RMSE
mean_theta_spice = Inf*ones(1,N_var);
mean_theta_likes = Inf*ones(1,N_var);
mean_theta_slim  = Inf*ones(1,N_var);
mean_theta_iaa   = Inf*ones(1,N_var);

mean_theta_beam   = Inf*ones(1,N_var);

%TODO: Find all finite estimates, compute mean, take sqrt
for n = 1:N_var
    
    idx = find( isfinite(sq_theta_error_spice(1,n,:)) == 1 );
    mean_theta_spice(1,n) = mean(sq_theta_error_spice(1,n,idx)) / K_true;
    
    idx = find( isfinite(sq_theta_error_likes(1,n,:)) == 1 );
    mean_theta_likes(1,n) = mean(sq_theta_error_likes(1,n,idx)) / K_true;
    
    idx = find( isfinite(sq_theta_error_slim(1,n,:)) == 1 );
    mean_theta_slim(1,n) = mean(sq_theta_error_slim(1,n,idx)) / K_true;
    
    idx = find( isfinite(sq_theta_error_iaa(1,n,:)) == 1 );
    mean_theta_iaa(1,n) = mean(sq_theta_error_iaa(1,n,idx)) / K_true;
    
    idx = find( isfinite(sq_theta_error_beam(1,n,:)) == 1 );
    mean_theta_beam(1,n) = mean(sq_theta_error_beam(1,n,idx)) / K_true;

end



%% Plot

close all

% %NMSE - LMMSE
% figure(1)
% plot(N_dim_list, 10*log10(NMSE_spice), 'k+-','LineWidth', 1.3), hold on, grid on
% plot(N_dim_list, 10*log10(NMSE_likes), 'bo-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_slim), 'gd-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_iaa), 'rx-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_oracle), 'k--','LineWidth', 1.3)
% ylabel('MSE relative to signal power [dB]')
% xlabel('N')
% title('LMMSE')
% legend('SPICE','LIKES','SLIM','IAA','Oracle')
% 
% 
% %NMSE - Capon
% figure(2)
% plot(N_dim_list, 10*log10(NMSE_spice_lmvue), 'k+-','LineWidth', 1.3), hold on, grid on
% plot(N_dim_list, 10*log10(NMSE_likes_lmvue), 'bo-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_slim_lmvue), 'gd-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_iaa_lmvue), 'rx-','LineWidth', 1.3)
% plot(N_dim_list, 10*log10(NMSE_oracle), 'k--','LineWidth', 1.3)
% ylabel('MSE relative to signal power [dB]')
% xlabel('N')
% title('Capon')
% legend('SPICE','LIKES','SLIM','IAA','Oracle')


% %Pd
% figure(3),
% plot(N_dim_list, PrD_spice, 'k+-','LineWidth', 1.3), hold on, grid on
% plot(N_dim_list, PrD_likes, 'bo-','LineWidth', 1.3)
% plot(N_dim_list, PrD_slim, 'gd-','LineWidth', 1.3)
% plot(N_dim_list, PrD_iaa, 'rx-','LineWidth', 1.3)
% plot(N_dim_list, PrD_beam, 'k*--','LineWidth', 1.3)
% legend('SPICE','LIKES','SLIM','IAA','Beamformer')
% ylabel('P_d')
% xlabel('N')
% title('Probability of detection')
% 
% %RMSE
% figure(4)
% plot(N_dim_list, (180/pi)*sqrt(mean_theta_spice), 'k+-','LineWidth', 1.3), hold on, grid on
% plot(N_dim_list, (180/pi)*sqrt(mean_theta_likes), 'bo-','LineWidth', 1.3)
% plot(N_dim_list, (180/pi)*sqrt(mean_theta_slim), 'gd-','LineWidth', 1.3)
% plot(N_dim_list, (180/pi)*sqrt(mean_theta_iaa), 'rx-','LineWidth', 1.3)
% plot(N_dim_list, (180/pi)*sqrt(mean_theta_beam), 'k*--','LineWidth', 1.3)
% ylabel('RMSE [deg.]')
% xlabel('N')
% title('DOA')
% legend('SPICE','LIKES','SLIM','IAA','Beamformer')

%Clock
figure(5)
plot(N_dim_list, mean_clock_spice, 'k+-','LineWidth', 1.3), hold on, grid on
plot(N_dim_list, mean_clock_likes, 'bo-','LineWidth', 1.3)
plot(N_dim_list, mean_clock_slim, 'gd-','LineWidth', 1.3)
plot(N_dim_list, mean_clock_iaa, 'rx-','LineWidth', 1.3)
legend('SPICE','LIKES','SLIM','IAA')
ylabel('Average time [s]')
xlabel('N')
title('Computation time')


%Iterations
figure(6)
plot(N_dim_list, mean_iter_spice, 'k+-','LineWidth', 1.3), hold on, grid on
plot(N_dim_list, mean_iter_likes, 'bo-','LineWidth', 1.3)
plot(N_dim_list, mean_iter_slim, 'gd-','LineWidth', 1.3)
plot(N_dim_list, mean_iter_iaa, 'rx-','LineWidth', 1.3)
legend('SPICE','LIKES','SLIM','IAA')
ylabel('Average iterations')
xlabel('N')