clc;
clear;
close all;

% Define desired function
f = @(x) sin(0.2 * pi * x) .* exp(-x); % Example function
x = linspace(0, 1, 1000); % Domain
f_vals = f(x);

% Initialize neuron parameters
N = 10; % Number of neurons
x_k = linspace(0, 1, N); % Initial neuron positions
A_k = ones(1, N); % Initial amplitudes
alpha = .20; % Decay rate
eta = 0.1; % Learning rate

% Simulation parameters
num_iterations = 100;

% Error history
error_history = zeros(1, num_iterations);

% Training loop
for iter = 1:num_iterations
    % Compute network output
    hat_f_vals = zeros(size(x));
    for k = 1:N
        hat_f_vals = hat_f_vals + A_k(k) * exp(-alpha * abs(x - x_k(k)));
    end

    % Compute error
    error = f_vals - hat_f_vals;
    error_history(iter) = sum(error.^2);

    % Update neuron parameters
    for k = 1:N
        % Update positions
        dE_dxk = -2 * sum(error .* (A_k(k) * alpha * sign(x - x_k(k)) .* exp(-alpha * abs(x - x_k(k)))));
        x_k(k) = x_k(k) - eta * dE_dxk;

        % Update amplitudes
        dE_dAk = -2 * sum(error .* exp(-alpha * abs(x - x_k(k))));
        A_k(k) = A_k(k) - eta * dE_dAk;
    end
end

% Plot results
figure;
subplot(2, 1, 1);
plot(x, f_vals, 'b-', 'LineWidth', 2, 'DisplayName', 'Original Function');
hold on;
plot(x, hat_f_vals, 'r--', 'LineWidth', 2, 'DisplayName', 'Approximated Function');
scatter(x_k, zeros(size(x_k)), 50, 'k', 'filled', 'DisplayName', 'Neuron Positions');
legend show;
xlabel('x');
ylabel('f(x)');
title('Function Approximation with Spiked Exponentials');

subplot(2, 1, 2);
plot(1:num_iterations, error_history, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Error');
title('Error Over Time');
grid on;
