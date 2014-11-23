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


hXis = X * theta;
betaIs = hXis - y;


J = sum(betaIs .^2) / (2 * m);
J_regularization = (lambda / (2*m)) * sum(theta(2:size(theta)) .* theta(2:size(theta)));
J = J + J_regularization;




grads = X' * betaIs;
grad = grads / m;

% define a vector the same size as theta
% ones everywhere except in first position
mult_matrix = ones(size(theta));
mult_matrix(1) = 0;

grad_regularization = lambda / m * (theta .* mult_matrix);
grad = grad + grad_regularization;



% =========================================================================

grad = grad(:);

end