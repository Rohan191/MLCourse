function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Regularized cost function
h_X = X*theta;
reg = sum(theta(2:size(theta,1)) .^ 2) * lambda/ (2*m); %Caluclate regularized part
J = sum((h_X - y) .^ 2) / (2*m) + reg; %Cost function


%Regularized gradient
regGrad = zeros(size(theta));
regGrad(2:size(theta,1)) = theta(2:size(theta,1)) * lambda/m;
grad(1) = sum((h_X - y) .* X(:, 1)) / m;
for j=2:size(theta,1)
    grad(j) = sum((h_X - y) .* X(:, j)) / m + regGrad(j);

% =========================================================================

grad = grad(:);

end
