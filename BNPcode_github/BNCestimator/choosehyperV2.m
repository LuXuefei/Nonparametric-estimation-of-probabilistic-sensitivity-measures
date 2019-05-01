function [beta0, iC] = choosehyperV2(x,Y)
% Hyperparameters for the Normal-InverseGamma prior for the (beta,sigma^2)_j
k = convhull(x,Y);
a = nan(length(k)-1,1);b = nan(length(k)-1,1);
for i=1: length(k)-1
coefficients = polyfit([x(k(i)), x(k(i+1))], [Y(k(i)), Y(k(i+1))], 1);
a(i) = coefficients (1);
b(i) = coefficients (2);
end

coeffs = polyfit(x, Y, 1); %fit aX+b = Y, beta0 = [b,a]
beta0=[coeffs(2), coeffs(1)]';

rangeitc = max(abs(b-beta0(1)));
rangeslp = max(abs(a-beta0(2)));
%beta(slope)+3sigma>=max(slope)
iC=diag([(rangeitc/3)^2,  (rangeslp/3)^2]);
end