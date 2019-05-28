function [ A ] = func_ULA( theta, delta, M )
%% Create steering vectors
%Dave Zachariah

%Electrical angle
phi = -2*pi*delta * sin(theta);

%Matrix of steering vectors M x D
A = exp( 1j * (0:M-1)' * phi(:)' );


end

