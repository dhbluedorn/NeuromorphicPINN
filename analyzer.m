clc
clear
close all

F = readmatrix("Datahighsubstep.csv");



% Figure 1 Plots the system as a function of time

figure(1);
plot(F(:, 1), F(:, 2));
hold on;
grid on
plot(F(:, 1), F(:, 3));
title("Position-Time Plots");
xlabel("Simulation Time (ms)");
ylabel("Bob Position (meters)");

% Figure 3 Shows each bobs' motion in 2D

figure(2);
plot(F(:, 2), F(:, 3));
hold on
grid on
plot(F(:, 4), F(:, 5));
legend('Bob 1', 'Bob 2');

% Figure 3 Shows the Fourier transform of a single dimension
legendEntries = [];
figure(3);
hold on
grid on
for i = 1:length(F(1, :))-1
    y = fft(F(:, i+1) - mean(F(:, i+1)));
    Ts = mean(diff(F(:, 1))); % Average Time Step
    Fs = 1/Ts; % Average Frequency Step
    f = (0:length(y)-1)*Fs/length(y);
    plot(f,abs(y))
    xlabel('Frequency (Hz)')
    ylabel('Magnitude');
    title('Positional Fourier Transforms');
    legendEntries = [legendEntries, "Dimension " + num2str(i)];
end
legend(legendEntries);


% Angle Plot
figure(4);
angle1 = atan(F(:, 3)./F(:, 2));
plot(F(:, 1), angle1);
hold on
grid on
angle2 = atan(F(:, 5)./F(:, 4));
plot(F(:, 1), angle2);
title("Angular Bob Position");
xlabel("Time (ms)");
ylabel("Bob Angle (Radians)");
legend("Bob 1", "Bob 2");

% Relative Angle

figure(6);
r1 = [F(:, 2) F(:, 3)];
r2 = [F(:, 4), F(:, 5)];

rrel = r2 - r1;

angle3 = atan(rrel(:, 2) ./ rrel(:, 1));

plot(F(:, 1), angle3);
title("Terminal Bob Relative Angle");
xlabel("Time (ms)");
ylabel("Bob Relative Angle (Radians)");
grid on

% Angle Fourier Analysis
figure(5);
y2 = fft(angle1 - mean(angle1));
f2 = (0:length(y2)-1)*Fs/length(y2);
plot(f2,abs(y2));
hold on
grid on
y3 = fft(angle2 - mean(angle2));
f3 = (0:length(y3)-1)*Fs/length(y3);
plot(f3,abs(y3));
y4 = fft(angle3 - mean(angle3));
f4 = (0:length(y4)-1)*Fs/length(y4);
plot(f4,abs(y4));
title("Fourier Transform of Angular Data");
xlabel("Frequency (Hz)");
ylabel("Magnitude)");





% Analyze Plots

highestF = f(find(maxk(abs(y), 5)));

highestF2 = f2(find(maxk(abs(y2), 5)));

highestF3 = f3(find(maxk(abs(y4), 5)));



