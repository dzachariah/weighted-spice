clear all, close all, clc

%% Seed
%RandStream.setGlobalStream(RandStream('mt19937ar','seed',5));
RandStream.setGlobalStream(RandStream('mt19937ar','seed','shuffle'));

%% Set parameters
%MC parameters
MC_runs = 1;

%Array and grid dimensions
N_dim     = 35
M_grid    = 1000
%theta_set = linspace(-pi/2,pi/2,M_grid);

%Signal parameters
idx_true    = [400 420 600] %true support set
P_diag_true = [1 3 2].^2            %signal power distribution
SNR         = 20;                   %signal to noise-ratio [dB] TODO: define!
delta       = 1/2;                  %wavelength separation


%% Resulting parameters

%Unknown structures
%theta_true = theta_set(idx_true); % DOA
P_true    = diag(P_diag_true);
%A_true    = func_ULA( theta_true, delta, N_dim );
K_true    = length(P_diag_true);
%R_true = (A_true*P_true*A_true') + sigma2 * eye(N_dim);


%Fixed structures
I_N       = eye(N_dim);
%A_steer   = func_ULA( theta_set, delta, N_dim );
%A_aug     = [A_steer, I_N];


%% Algorithm parameters
%Common
conv_tol      = 1e-3; %tol |p^{i+1} - p^{i}|/|p^{i}| at each step
N_powerupdate = 1;


%% Allocate memory to store

%Power spectrum
x_hat_spice = zeros(MC_runs,M_grid);
x_hat_likes = zeros(MC_runs,M_grid);
x_hat_slim  = zeros(MC_runs,M_grid);
x_hat_iaa   = zeros(MC_runs,M_grid);

%Iterations
iter_spice = zeros(MC_runs,1);
iter_likes = zeros(MC_runs,1);
iter_slim  = zeros(MC_runs,1);
iter_iaa   = zeros(MC_runs,1);



%% Monte Carlo
T_snap = 1;

%TODO: Plot amplitudes instead

tic
for m = 1:MC_runs
    
    %Generate signals
    %---------------------------
    %Generate SOI (complex Gaussian)
    x_true_sub         = chol(P_true)' * exp(1i*2*pi*rand(K_true, T_snap)); %random phase
    x_true             = zeros(M_grid,1);
    x_true(idx_true,1) = x_true_sub;
    
    %Construct noise covariance matrix
    sigma2       = (trace(P_true)) /( (trace(I_N)/N_dim) * 10^(SNR/10) ); %TODO: Note definition
        
    %Generate noise
    n_true = sqrt(sigma2) * (randn(N_dim,T_snap) + 1i*randn(N_dim,T_snap))/sqrt(2);
    
    %Generate regressor
    A_true    = sqrt(1) * (randn(N_dim,M_grid) + 1i*randn(N_dim,M_grid))/sqrt(2);
    A_aug     = [A_true, I_N];
    
    %Generate array output
    y = (A_true*x_true) + n_true;

    %Amplitude estimates
    %---------------------------
    %SPICE
    disp('SPICE-1')
    type_algorithm = 1; flag_version     = 0; N_powerupdate = 1; iter_limit = 1e3;
    [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );
    
    %idx_hat_spice              = func_modelselector( real(p_tmp(1:M_grid)), K_max );
    x_hat_spice                = func_lmmse( y, A_aug, p_tmp );
    x_hat_spice_lmvue          = func_lmvue( y, A_aug, p_tmp );
    %iter_spice(1,count,m)      = iter_tmp;
    %clock_spice(1,count,m)     = clock_tmp;
    p_hat_spice                = p_tmp; %for likes
    
    
    %LIKES
    disp('SPICE-2')
    type_algorithm = 2; flag_version     = 0; N_powerupdate = 30; iter_limit = 1e3;
    [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, p_hat_spice );
    
    %idx_hat_likes              = func_modelselector( real(p_tmp(1:M_grid)), K_max );
    x_hat_likes                = func_lmmse( y, A_aug, p_tmp );
    x_hat_likes_lmvue          = func_lmvue( y, A_aug, p_tmp );
    %iter_likes(1,count,m)      = iter_tmp;
    %clock_likes(1,count,m)     = clock_tmp + clock_spice(1,count,m); %total time
    
    
    %SLIM
    disp('SPICE-3')
    type_algorithm = 3; flag_version     = 0; N_powerupdate = 1; iter_limit = 5;
    %[p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );
    [p_tmp,iter_tmp,clock_tmp] = func_iaa_a( y, A_aug, conv_tol, flag_version, 1e3 );
    
    %idx_hat_slim               = func_modelselector( real(p_tmp(1:M_grid)), K_max );
    x_hat_slim                 = func_lmmse( y, A_aug, p_tmp );
    x_hat_slim_lmvue           = func_lmvue( y, A_aug, p_tmp );
    %iter_slim(1,count,m)       = iter_tmp;
    %clock_slim(1,count,m)      = clock_tmp;
    
    
    %IAA
    disp('SPICE-4')
    type_algorithm = 4; flag_version     = 0; N_powerupdate = 1; iter_limit = 1e3;
    [p_tmp,iter_tmp,clock_tmp] = func_spice_unified( y, A_aug, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, [] );
    
    %idx_hat_iaa                = func_modelselector( real(p_tmp(1:M_grid)), K_max );
    x_hat_iaa                  = func_lmmse( y, A_aug, p_tmp );
    x_hat_iaa_lmvue            = func_lmvue( y, A_aug, p_tmp );
    %iter_iaa(1,count,m)        = iter_tmp;
    %clock_iaa(1,count,m)       = clock_tmp;
    
    
end
toc

%% Save data


%% Compute statistics

%Compute mean
amp_spice_mean = mean( abs(x_hat_spice), 2 );
amp_likes_mean = mean( abs(x_hat_likes), 2 );
amp_slim_mean  = mean( abs(x_hat_slim), 2 );
amp_iaa_mean   = mean( abs(x_hat_iaa), 2 );


%% Plot

%Plot mean
figure(1)
subplot(2,2,1)
semilogy( 1:M_grid, amp_spice_mean, 'b-', 'LineWidth', 1.2 ), hold on, grid on
semilogy( idx_true, sqrt(P_diag_true), 'kx', 'LineWidth', 1.5 )
legend('SPICE','true'), xlabel('index $k$'), ylabel('$|\hat{x}|$')
axis([1 M_grid  1e-4 5])

%figure(2)
subplot(2,2,2)
semilogy( 1:M_grid, amp_likes_mean, 'b-', 'LineWidth', 1.2 ), hold on, grid on
semilogy( idx_true, sqrt(P_diag_true), 'kx', 'LineWidth', 1.5 )
legend('LIKES','true'), xlabel('index $k$'), ylabel('$|\hat{x}|$')
axis([1 M_grid  1e-4 5])

%figure(3)
subplot(2,2,3)
semilogy( 1:M_grid, amp_slim_mean, 'b-', 'LineWidth', 1.2 ), hold on, grid on
semilogy( idx_true, sqrt(P_diag_true), 'kx', 'LineWidth', 1.5 )
legend('SLIM','true'), xlabel('index $k$'), ylabel('$|\hat{x}|$')
axis([1 M_grid  1e-4 5])

%figure(4)
subplot(2,2,4)
semilogy( 1:M_grid, amp_iaa_mean, 'b-', 'LineWidth', 1.2 ), hold on, grid on
semilogy( idx_true, sqrt(P_diag_true), 'kx', 'LineWidth', 1.5 )
legend('IAA','true'), xlabel('index $k$'), ylabel('$|\hat{x}|$')
axis([1 M_grid  1e-4 5])
