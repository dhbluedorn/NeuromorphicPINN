% Physics Identification Unit (PIU) Demo

clc
clear
close all

% Load trained neural structure
load neurons.mat neurons

% Define Linear damped oscillator

dt = 0.01;

t = 1:dt:50;

M = 7;
K = 2;
b = .2;
x = exp(-b*t/(2*M)) .* sin(((sqrt(4*M*K)-b^2)/(2*M))*t);

% Input data
inputSeries = x;

% Initial plot of default network output
plotPut = [];
for i = 1:length(inputSeries)
    input = inputSeries(i);
    [output, neurons] = processNetwork(input, neurons);
    plotPut = [plotPut, output];
end

figure(2);
subplot(2, 1, 1);
plot(plotPut(4:end));
hold on
grid on
hold off
xlabel('Time');
ylabel('Output');
title('Network Output');

% Perform Synthetic Recall
for i = 1:length(neurons)
    for j = 1:length(neurons{i})
        weightMatrix(i, j) = neurons{i}(j).weight;
    end
end
w = weightMatrix';
Mnet = w(3, 1)
bnet = w(2, 1)
Knet = w(1, 1)

x2 = exp(-bnet*t/(2*Mnet)) .* sin(((sqrt(4*Mnet*Knet)-bnet^2)/(2*Mnet))*t);

x = exp(-b*t/(2*M)) .* sin(((sqrt(4*M*K)-b^2)/(2*M))*t);
figure(2);
subplot(2, 1, 2);
plot(t, x);
hold on
plot(t, x2);
grid on
title("Original System Physics");
xlabel("Time (seconds)");
ylabel("Displacement (meters)");
legend("Original", "PIU");

fprintf("MSE Loss over extrapolation domain is %2.2f%%\n", 100*(1/length(x)) * sum((x - x2).^2));
% Forward pass for the network
function [output, neurons] = processNetwork(input, neurons)
    layerOutput = input;
    for layer = 1:length(neurons)
        nextLayerOutput = 0;
        for i = 1:length(neurons{layer})
            [neuronOutput, neurons{layer}(i)] = neurons{layer}(i).processInput(layerOutput); % Update each neuron in the array
            nextLayerOutput = nextLayerOutput + neuronOutput;
        end
        layerOutput = nextLayerOutput;
    end
    output = layerOutput;
end