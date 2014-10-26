function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        % disp('')
        % disp('==========')
        % disp(sprintf('Iteration %d', iter));
        % theta

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %

        % V1
        % sum_part_1 = 0.0;
        % sum_part_2 = 0.0;

        % for i = 1:m
        %     prediction = [1, X(i,2)] * theta - y(i);
        %     sum_part_1 +=
        % end;

        % sum_part_2 = 0.0;
        % for i = 1:m
        %     prediction = [1, X(i,2)] * theta;
        %     sum_part_1 += (prediction - y(i)) * X(i,2);
        % end;

        % theta(1) = theta(1) - alpha/m * sum_part_1;
        % theta(2) = theta(2) - alpha/m * sum_part_2;


        % V2    
        delta = [ 0; 0 ];
        delta(1, 1) = (sum(((X * theta) - y) .* X(:,1))) / m;
        delta(2, 1) = (sum(((X * theta) - y) .* X(:,2))) / m;
        theta = theta - alpha * delta;

        % ============================================================

        % Save the cost J in every iteration    
        cost = computeCost(X, y, theta);
        J_history(iter) = cost;

    end

end
