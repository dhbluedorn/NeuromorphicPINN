% DifferentialNeuron Class
classdef DifferentialNeuron
    properties
        weight % Weight associated with this neuron
        memory % Vector to store previous inputs
        dt % Time step size for differencing
        order % Order of the derivative
        lastOutput % Store last output for gradient calculation
    end
    
    methods
        function obj = DifferentialNeuron(order, weight, dt, memorySize)
            obj.order = order;
            obj.weight = weight;
            obj.dt = dt;
            obj.memory = [0 0 0]; % Initialize memory buffer with NaNs
            obj.lastOutput = 0;
        end
        
        function [output, obj] = processInput(obj, input)
            % Shift memory and store new input
            obj.memory = [input, obj.memory(1:end-1)];

            % Calculate finite difference based on the order
            if obj.order == 0
                output = input * obj.weight;
            elseif obj.order == 1
                % First-order derivative
                output = ((obj.memory(1) - obj.memory(2)) / obj.dt) * obj.weight;
            elseif obj.order == 2
                % Second-order derivative
                output = ((obj.memory(1) - 2*obj.memory(2) + obj.memory(3)) / (obj.dt^2)) * obj.weight;
            else
                output = 0; % Return zero if unsupported order
            end
      
            obj.lastOutput = output; % Store output for gradient calculation
        end
            
        function obj = updateWeight(obj, learningRate, lossGradient)
            % Gradient update rule
            obj.weight = obj.weight - learningRate * lossGradient * obj.lastOutput;
        end
    end
end
        
        

