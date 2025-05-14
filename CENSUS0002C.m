% CENSUS 0002-C - Error Function Refinement
% Daniel Bluedorn - Neuromorphic PINN Project

% CENSUS 0002 Control Unit

clc
clear
close all


video_filename = 'SlimeNetAnimation.mp4';
v = VideoWriter(video_filename, "MPEG-4");
v.FrameRate = 20; % Set the frame rate
open(v);


% ----------------- SlimeNet Model Function -----------------
function tau = SlimeNetController(theta, tau, dt, command)
    persistent positions velocities thresholds intensities active_times ...
        domain_size num_neurons input_position output_position ...
        bthres bsig eta step alpha t output_intensity prev_intensity
    if nargin > 3
        switch command
            case 'save'
                save('SlimeNetState.mat', 'positions', 'velocities', 'thresholds', ...
                     'intensities', 'active_times', 'domain_size', 'num_neurons', ...
                     'input_position', 'output_position', 'bthres', 'bsig', ...
                     'eta', 'step', 'alpha', 't', 'output_intensity', 'prev_intensity');
                disp('Persistent variables saved.');
                return;
            case 'load'
                if isfile('SlimeNetState.mat')
                    load('SlimeNetState.mat', 'positions', 'velocities', 'thresholds', ...
                         'intensities', 'active_times', 'domain_size', 'num_neurons', ...
                         'input_position', 'output_position', 'bthres', 'bsig', ...
                         'eta', 'step', 'alpha', 't', 'output_intensity', 'prev_intensity');
                    disp('Persistent variables loaded.');
                else
                    disp('No saved state found. Initializing new variables.');
                end
                return;
        end
    end

    if isempty(positions)
        % Initialize SlimeNet parameters
        num_neurons = 30;        % Number of neurons
        domain_size = 200;       % 1D domain length
        eta = .05;                 % Learning rate
        bthres = 150;            % Threshold decay rate
        bsig = 300;              % Signal damping factor
        alpha = 0.01;             % Smoothing factor
        step = 0;

        % Initialize Neuron State
        positions = linspace(0.2*domain_size, 0.8*domain_size, num_neurons)';
        velocities = zeros(num_neurons, 1);
        thresholds = ones(num_neurons, 1);
        intensities = zeros(num_neurons, 1);
        active_times = zeros(num_neurons, 1);

        % Input and output positions
        input_position = 0;           % Input at the leftmost point
        output_position = domain_size; % Output at the rightmost point
        output_intensity = 0;
        prev_intensity = 0;
    end

    substeps = 50;
    subDt = dt/substeps;
    % External signal based on pendulum state (feedback control)
    input_signal = min(abs(theta), 1); % Error signal based on theta & omega

    
    for j = 1:substeps
        % Update Neuron States
        forces = zeros(num_neurons, 1);
        output_intensity = 0;
        t = step * subDt;
        for i = 1:num_neurons
            % Input influence
            dist_to_input = abs(positions(i) - input_position);
            if dist_to_input > 1e-6
                intensities(i) = intensities(i) + input_signal / dist_to_input;
            end
    
            % Contribution to output
            dist_to_output = abs(positions(i) - output_position);
            if dist_to_output > 1e-6
                output_intensity = output_intensity + intensities(i) / dist_to_output;
            end
    
            % Total intensity at neuron
            I = sum(intensities(1:i-1)./abs(positions(1:i-1) - positions(i)) + input_signal);
    
            % Check neuron activation
            if I > thresholds(i)
                active_times(i) = t;
                thresholds(i) = 1;
                intensities(i) = 1;
            end
    
            dEdx = zeros(num_neurons, 1);
            for i = 1:num_neurons
                dist_to_output = abs(positions(i) - output_position);
                if (dist_to_output > 3)
                    dEdx(i) = 2 * (abs(theta^2 + tau^2)) * (intensities(i) / dist_to_output);
                end
            end
            if max(positions - eta * dEdx) > 0
                positions = positions - eta * dEdx;
            end  
        end
         % Decay thresholds and intensities
        thresholds = thresholds .* exp(-bthres * (t - active_times));
        intensities = intensities .* exp(-bsig * (t - active_times));
    
        % Exponential smoothing for output intensity
        output_intensity = alpha * output_intensity + (1 - alpha) * prev_intensity;
        if output_intensity > 1
            output_intensity = 1;
        end
        prev_intensity = output_intensity;
    
        % Torque Decision
        tau = -sign(theta) * output_intensity;
    end
    step = step + 1;

   
end

% ----------------- Inverted Pendulum Simulation -----------------
% 1. Initial Conditions
%SlimeNetController([], [], [], 'load');
theta0 = (180/180)*pi;
omega0 = 0;
alpha0 = 0;

L = 1;            
g = 9.81;         
dt = .01;
theta = theta0;   
omega = omega0;   
tau = 0;          
tau2 = 0;
tau3 = 0;
maxTorque = 30;

% 2. Simulation Loop
T = [];
TAU = [tau; tau2];
THETA = [];
OMEGA = [];
Q = [];

% Define Desired Trajectory

tss = 0.55;

es = 0.05;

b = -log(es)/tss;


for t = 0:dt:1.4
    % Define Desired Trajectory

    q = theta0 * [exp(-b*t) -b*exp(-b*t) b^2*exp(-b*t)]; % Desired motion quaternion
    Q = [Q; q];

    % Calculate angular acceleration (equation of motion)
    alpha = (g/L)*sin(theta) + (tau + tau2 + tau3)/(L^2);

    % Update angular velocity and angle
    omega = omega + alpha * dt;
    theta = theta + omega * dt;

    % Call SlimeNet to get the control torque
    controlRatio = 0.69; % Ratio of Oscillator torque to damper torque
    priorityRatio = 1;
    tau = maxTorque * SlimeNetController(theta-q(1), tau, dt);
    tau2 = maxTorque * SlimeNetController(omega-q(2), tau2, dt);

    T = [T tau+tau2];
    THETA = [THETA theta];
    OMEGA = [OMEGA omega];
    TAU = [TAU [tau; tau2]];
    

    % Clear the figure
    clf;

    % Subplot 1: Pendulum animation
    subplot(3, 1, 1);
    hold on;
    grid on;
    axis equal;
    xlim([-4.0 4.0]);
    ylim([-1.1 1.1]);

    % Draw pendulum
    plot([0 L*sin(theta)], [0 L*cos(theta)], 'r', 'LineWidth', 4); % Rod
    plot(L*sin(theta), L*cos(theta), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % Bob
    plot(0, 0, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b'); % Pivot

    title(['Time: ', num2str(t, '%.2f'), ' s | Torque: ', num2str(round(tau+tau2+tau3, 1)), 'Nm | Theta: ', num2str(round(theta*180/pi, 1)), '° | Omega: ', num2str(round(omega, 1))]);
    xlabel('X');
    ylabel('Y');

    % Subplot 2: Torque plot
    subplot(3, 1, 2);
    plot(0:dt:t, T, 'b', 'LineWidth', 1.5);
    hold on
    plot(t, tau+tau2, 'bo', 'LineWidth',1.5);
    title('Motor Torque (N*m)');
    ylabel('Torque');
    ylim([-2*maxTorque 2*maxTorque]);
    grid on;

    % Subplot 3: Theta plot
    subplot(3, 1, 3);
    plot(0:dt:t, THETA*180/pi, 'r', 'LineWidth', 1.5);
    hold on
    plot(t, theta*180/pi, 'rx', 'LineWidth', 1.5);
    plot(0:dt:t, Q(:, 1), 'b', 'LineWidth', 1.5);
    title('Pendulum Response');
    xlabel('Time Step');
    ylabel('Deviation (°)');
    ylim([-theta0*180/pi theta0*180/pi]);
    grid on;
    % Update the figure
    drawnow;
    % Capture the frame and write to video
    frame = getframe(gcf);
    writeVideo(v, frame);

    % Stop if pendulum falls over
    %{
    if abs(theta) > pi/2
        disp('Pendulum fell over!');
        break;
    end
    %}
end
close(v);
%% Post-Processing
audio = T ./ max(T);
save audio

SlimeNetController([], [], [], 'save');

time = 0:dt:t;

ts = findSettlingTime(THETA, theta0, 0.05, 0:dt:t);

fprintf("The system settles to equilibrium after <strong>%2.2f seconds</strong>\n", ts);

figure(3);
hold on
grid on
plot(Q(:, 1), 'LineWidth', 1.5);
plot(Q(:, 2), 'LineWidth', 1.5);
plot(Q(:, 3), 'LineWidth', 1.5);
plot(THETA, 'LineWidth', 1.5);
legend("Theta (rad)", "Omega (rad/s)", "Alpha (rad/s^2)", "Theta Actual");
title("Desired System Quaternion");
