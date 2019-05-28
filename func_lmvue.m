function [ x_hat ] = func_lmvue( y, A, p )
%% Compute LMVUE/Capon estimate

%% Initialize
[N,N_tot] = size(A);
M         = N_tot - N;
I_N       = eye(N);

%% LMVUE
R_inv   = (A*diag(p)*A') \ I_N;
y_tilde = R_inv * y; 
x_hat   = zeros(M,1);

for k = 1:M
    x_hat(k) = A(:,k)'*y_tilde / ( A(:,k)'*R_inv*A(:,k) );
end


end
