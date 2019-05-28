function [ x_hat ] = func_lmmse( y, A, p )
%% Compute LMMSE estimate

%% Initialize
[N,N_tot] = size(A);
M         = N_tot - N;
I_N       = eye(N);

%% LMMSE
R_inv   = (A*diag(p)*A') \ I_N;
y_tilde = R_inv * y; 
x_hat   = zeros(M,1);

for k = 1:M
    x_hat(k) = p(k) * A(:,k)'*y_tilde;
end


end
