function [p_hat,iter,clock_time] = func_spice_unified( y, A, conv_tol, type_algorithm, flag_version, N_powerupdate, iter_limit, p_hat_old )
% Unified SPICE
% Dave Zachariah 2013-09-16
% dave.zachariah@it.uu.se
% Please cite if used
%% Input/Output
%Input:
%------------------
% y              - data vector N x 1
% A              - regressor matrix NxM
% type_algorithm - 1 = SPICE, 2 = LIKES, 3 = SLIM, 4 = IAA
% flag_version   - 0 = A-type, 1 = B-type gradient step length
% N_powerupdate  - # power updates for LIKES, SLIM and IAA
% iter_limit     - iteration upper limit
% p_hat_old      - initial power estimates
%Output:
%------------------
% p_hat          - power estimate ([] is default input)
% iter           - number of iterations (outer loop)
% clock_time     - total CPU time

%% Global variables and start clock
tic

global N_dim
global N_tot
global M
global I_N
global R_inv


%% Set variables
[N_dim,N_tot] = size(A);
p_hat         = zeros(1,N_tot);
w             = zeros(1,N_tot);
M             = N_tot - N_dim;
I_N           = eye(N_dim);
iter          = 0; %iteration counter

%Repeated updates
if type_algorithm == 1
    N_repeat = N_powerupdate;
else
    N_repeat = 1;
end


%% Initialize
%Weights
for k = 1:N_tot
    w(k) = norm( A(:,k) )^2;
end

%(Powers)
if isempty(p_hat_old)
    p_hat_old = zeros(1,N_tot);
    for k = 1:N_tot
        p_hat_old(k) = abs( A(:,k)'*y/w(k) )^2; %Periodigram-style
    end
end

    
%% Iterate

while(true)
 
    %Covariance update
    %--------------------
    R_inv   = ( A*diag(p_hat_old)*A' ) \ I_N; %TODO: Faster implementation using 'fun'?
    y_tilde = R_inv * y;

    %(Update weights)
    %--------------------
    if  type_algorithm ~= 1
        %LIKES, SLIM & IAA
        if mod(iter,N_powerupdate) == 0
            [w] = func_weightupdate(p_hat_old, A, w, type_algorithm);
        end
    end
    
    
    %Power update
    %--------------------
    for j = 1:N_repeat
        p_hat = func_powerupdate(p_hat_old, y_tilde, A, w, flag_version);
    end
 
    %Convergence
    %--------------------    
    %DISP:
    %disp(norm(p_hat-p_hat_old)/norm(p_hat_old))
    %disp(iter)
    
    %Terminate
    if (norm(p_hat-p_hat_old)/norm(p_hat_old) < conv_tol) || (iter > iter_limit)
        if (iter > iter_limit)
            iter = inf;
            disp('Terminated before convergence')
        end
        break
    else
        p_hat_old = p_hat;
        iter      = iter + 1;
    end 

    
end


%% Check clock and exit
clock_time = toc;

end

%-----------------------
% Additional functions
%-----------------------


%Weight update
%-----------------------
function [w] = func_weightupdate(p_hat, A, w, type_algorithm)

global N_tot
global R_inv


if type_algorithm == 1
    %SPICE
    %constant weights

elseif type_algorithm == 2
    %LIKES
    for k = 1:N_tot
        w(k) = real(A(:,k)'*R_inv*A(:,k)); 
    end

elseif type_algorithm == 3
    %SLIM
    w = 1./p_hat;
    
elseif type_algorithm == 4
    %IAA
    for k = 1:N_tot
        w(k) = p_hat(k) * real(A(:,k)'*R_inv*A(:,k))^2; 
    end
    
else
    %ERROR
    disp('Error')
    pause
end
    
    
end


%Power update
%-----------------------
function [p_hat] = func_powerupdate(p_hat, y_tilde, A, w, flag_version)

global N_tot

if flag_version == 0
    %Version A
    for k = 1:N_tot
        p_hat(k) = p_hat(k) * abs( A(:,k)' * y_tilde ) / sqrt(w(k));
    end
    
elseif flag_version == 1
    %Version B
    for k = 1:N_tot
        p_hat(k) = p_hat(k) * abs( A(:,k)' * y_tilde )^2 / w(k);
    end
    
else
    %ERROR
    disp('Error')
    pause
end
    
    
end


