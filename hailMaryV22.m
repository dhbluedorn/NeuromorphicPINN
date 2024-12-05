% Version 2.2 - Network Normalization 
% Daniel Bluedorn - 11/24/2024

clc
clear
close all

% Parameters
dt = 0.01; % Time step
memorySize = 3; % Memory size for neurons
learningRate = 0.9;
epochs = 110;
scale = 1;
minStartWeight = 0.3;
t = 1:dt:20;

% Initialize network (layers of neurons, with bypass capability)
neurons = {[DifferentialNeuron(0, rand(), dt, memorySize) DifferentialNeuron(1, rand(), dt, memorySize) DifferentialNeuron(2, rand(), dt, memorySize)]};

M = 7;
K = 2;
b = .2;
x = exp(-b*t/(2*M)) .* sin(((sqrt(4*M*K)-b^2)/(2*M))*t);

% Input data
inputSeries = x;
targetSeries = zeros(size(inputSeries));

% Initial plot of default network output
plotPut = [];
for t = 1:length(inputSeries)
    input = inputSeries(t);
    [output, neurons] = processNetwork(input, neurons);
    plotPut = [plotPut, output];
end

figure(2);
subplot(3, 1, 1);
plot(plotPut(4:end));
hold on
grid on
hold off
xlabel('Time');
ylabel('Output');
title('Default Network Output');

% Training loop
[neurons, lossHistory] = trainNetwork(neurons, inputSeries, targetSeries, learningRate, epochs);

% Extract weights into a matrix after training

%%
for i = 1:length(neurons)
    for j = 1:length(neurons{i})
        weightMatrix(i, j) = neurons{i}(j).weight;
    end
end
w = weightMatrix';
networkSize = size(w);
% Perform Synthetic Induced Recall Algorithm
C = w(:, 1);
mod = 1;
for k = 1:networkSize(1)
    for j = 2:networkSize(2)
        mod = sum(w(1, j))*mod;
    end
    C(k) = C(k)*mod;
end

fprintf("The SIR matrix is: \n");
fprintf("C1: %1.4f\n", C(1));
fprintf("C2: %1.4f\n", C(2));
fprintf("C3: %1.4f\n", C(3));
t = 1:dt:100;
x2 = exp(-C(2)*t/(2*C(3))) .* sin(((sqrt(4*C(3)*C(1))-C(2)^2)/(2*C(3)))*t);

x = exp(-b*t/(2*M)) .* sin(((sqrt(4*M*K)-b^2)/(2*M))*t);
figure(2);
subplot(3, 1, 2);
hold on
plot(t, x);
plot(t, x2);
grid on
title("Original System Physics");
xlabel("Time (seconds)");
ylabel("Displacement (meters)");
legend("Original", "PIU");

% Plot training loss
figure(3);
plot(lossHistory(4:end));
hold on
grid on
hold off
xlabel('Epoch');
ylabel('MSE Loss');
title('Training Results');

% Testing Loop
plotPut = [];
for t = 1:length(inputSeries)
    input = inputSeries(t);
    [output, neurons] = processNetwork(input, neurons);
    plotPut = [plotPut, output];
end

figure(2);
subplot(3, 1, 3);
plot(plotPut(4:end));
hold on
grid on
xlabel('Time');
ylabel('Output');
title('Trained Output');

% Export Network Parameters

save('neurons.mat', 'neurons');

fprintf("MSE Loss over extrapolation domain is %2.2f%%\n", 100*(1/length(x)) * sum((x - x2).^2));


%% Function Definitions

% Training Function
function [neurons, lossHistory] = trainNetwork(neurons, inputSeries, targetSeries, learningRate, epochs)
    lossHistory = zeros(1, epochs);
    
    for epoch = 1:epochs
        totalLoss = 0;
        
        for t = 1:length(inputSeries)
            input = inputSeries(t);
            target = targetSeries(t);
            
            % Forward pass
            [output, neurons] = processNetwork(input, neurons);
            
            % Calculate loss
            loss = (output - target)^2;
            totalLoss = totalLoss + loss;
            
            % Backpropagate the error, layer by layer
            error = 2 * (output - target);
            gradientThreshold = .1;
            
            % For each layer in reverse order
            for layer = length(neurons):-1:1
                for i = 1:length(neurons{layer})
                    neuron = neurons{layer}(i);
                    
                    % Compute gradient with tanh derivative
                    gradient = error * (1 - neuron.lastOutput^2) * neuron.lastOutput;
                    
                    % Apply gradient clipping
                    
                    if abs(gradient) > gradientThreshold
                        gradient = sign(gradient) * gradientThreshold;
                    end
                    
                    % Update weight
                    if ((neuron.weight - learningRate * gradient) >= 0) && ((neuron.weight - learningRate * gradient) <= 1)
                        neuron.weight = neuron.weight - learningRate * gradient;
                    end

            
                    % Update the error to propagate backward to previous layers
                    error = error * neuron.weight * (1 - neuron.lastOutput^2);
            
                    neurons{layer}(i) = neuron; % Save updated neuron
                end
            end
        end
        
        % Store loss per epoch
        lossHistory(epoch) = totalLoss / length(inputSeries);
        fprintf('Epoch %d, Loss: %.4f\n', epoch, lossHistory(epoch));
    end
end

% Forward pass for the network
function [output, neurons] = processNetwork(input, neurons)
    layerOutput = input;
    for layer = 1:length(neurons)
        nextLayerOutput = 0;
        for i = 1:length(neurons{layer})
            [neuronOutput, neurons{layer}(i)] = neurons{layer}(i).processInput(layerOutput); % Update each neuron in the array
            nextLayerOutput = nextLayerOutput + tanh(neuronOutput);
        end
        layerOutput = nextLayerOutput;
    end
    output = layerOutput;
end
