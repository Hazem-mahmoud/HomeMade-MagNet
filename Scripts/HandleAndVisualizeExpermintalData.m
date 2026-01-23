%% --- 1. User Inputs (Replace these with your actual values) ---
ExperimentID=1;
V_sec = Data.Voltage(ExperimentID,:);  % Your voltage array (Volts)
I_prim = Data.Current(ExperimentID,:); % Current into Primary (Amps)
N_prim = Data.Primary_Turns;     % Turns on the current-carrying winding (Primary)
N_sec = Data.Secondary_Turns;      % Turns on the voltage-sensing winding (Secondary)
Ae = Data.Effective_Area;            % Cross-sectional area of the core (m^2)
Le = Data.Effective_Length;     % Effective Path Length (meters) -> e.g., 50mm
dt = Data.Sampling_Time(ExperimentID);              % Time step (seconds)
Hdc=Data.Hdc_command(ExperimentID);
Temperature = Data.Temperature_command(ExperimentID);
DutyP = Data.DutyP_command(ExperimentID);
DutyN = Data.DutyN_command(ExperimentID);
Frequency = Data.Frequency_command(ExperimentID);
Flux_cmd = Data.Flux_command(ExperimentID);

Material = Data.Material;
Shape = Data.Shape;
Ve = Data.Effective_Volume;
CoreN = Data.CoreN;
Dataset = Data.Dataset;

% Info Strings
Discarding_info = Data.Discarding_info;
Freq_info = Data.Freq_info;
Cycle_info = Data.Cycle_info;
Date_processing = Data.Date_processing;

% Create time vector (optional, for plotting)
t = (0:length(V_sec)-1) * dt;

% --- 2. Calculate Magnetizing Force (H) ---
% H = (N * I) / Le
H = (N_prim .* I_prim) ./ Le;

% --- 3. Calculate Flux Density (B) ---
% Remove DC offset to prevent integration drift
%V_clean = V_sec - mean(V_sec);

% Integrate Voltage: Integral(V) dt
Flux = cumtrapz(t, V_sec);


% B = Flux / (N * Ae)
B = Flux ./ (N_sec * Ae);
B = B - mean(B);

Energy_Per_Cycle = polyarea(H, B); 

% Power Loss Density (W/m^3) = Energy * Frequency
Pv = Energy_Per_Cycle * Frequency;



% --- 4. Plotting ---
figure('Name', 'B-H Analysis', 'Color', 'white');

% Subplot 1: Time Domain Signals
% Subplot 1: Voltage and Current (Time Domain)
subplot(2,2,1); % Top-Left
yyaxis left
plot(t, V_sec, 'LineWidth', 1.5);
ylabel('Voltage (V)');
xlabel('Time (s)');
ylim auto

yyaxis right
plot(t, I_prim, 'LineWidth', 1.5);
ylabel('Current (A)');
title('Voltage & Current');
legend('V (Volts)', 'I (Amps)');
grid on;

% Subplot 2: B and H (Time Domain)
subplot(2,2,3); % Bottom-Left
yyaxis left
plot(t, B, 'LineWidth', 1.5);
ylabel('Flux Density B (Tesla)');
xlabel('Time (s)');
ylim auto

yyaxis right
plot(t, H, 'LineWidth', 1.5);
ylabel('Field Strength H (A/m)');
title('B & H Waveforms');
legend('B (Tesla)', 'H (A/m)');
grid on;


% Subplot 2: B-H Loop (Hysteresis)
subplot(2,2,[2 4]); % Right side
plot(H, B, 'LineWidth', 2);
xlabel('Magnetizing Force H (A/m)');
ylabel('Flux Density B (Tesla)');
title('B-H Hysteresis Loop');
grid on;
axis tight;

% Add Info Box to Subplot 2 (Top-Left)
InfoString = {
    ['Hdc: ', num2str(Hdc), 'A/m'];
    ['Temp: ', num2str(Temperature), '°C'];
    ['Freq: ', num2str(round(Frequency/1000)), 'kHz'];
    ['Duty: ', num2str(DutyP)];
    ['Flux Cmd: ', num2str(Flux_cmd, '%.3f'), 'T']
    };
text(0.05, 0.95, InfoString, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

%% % --- 0. Load Data (if not in workspace) ---
% FileName = '3C90_TX-25-15-10_Data1_Cycle.mat';
% if ~exist('Data', 'var')
%     if exist(FileName, 'file')
%         fprintf('Loading %s...\n', FileName);
%         load(FileName);
%     else
%         error('Data structure not found and file %s missing.', FileName);
%     end
% end
% 
% % --- 0.5 Batch Identification: Sinusoidal Excitation ---
% fprintf('Running Sinusoidal Identification on %d experiments...\n', size(Data.Voltage, 1));

% Margin for RMS ~= Peak/sqrt(2)
% Let's use 5% tolerance
SineMargin = 0.005; 

[NumExp, ~] = size(Data.Voltage);
Data.Sinusoidal = zeros(NumExp, 1);

for k = 1:NumExp
    v_sig = Data.Voltage(k, :);
    i_sig = Data.Current(k, :);
    
    % --- FFT Checker Function ---
    % Checks if >90% of AC power is in the fundamental frequency
    check_spectral_purity = @(sig) calculate_purity(sig);
    
    is_v_sine = check_spectral_purity(v_sig);
    is_i_sine = check_spectral_purity(i_sig);
    
    % If either is sinusoidal, mark as 1
    if is_v_sine || is_i_sine
        Data.Sinusoidal(k, 1) = 1;
    else
        Data.Sinusoidal(k, 1) = 0;
    end
end

function is_pure = calculate_purity(sig)
    % Remove DC
    sig_ac = sig - mean(sig);
    L = length(sig_ac);
    
    if max(abs(sig_ac)) < 1e-9
        is_pure = false;
        return;
    end
    
    % FFT
    Y = fft(sig_ac);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    PowerSpec = P1.^2;
    
    % Find Fundamental (Max Peak)
    [max_p, ~] = max(PowerSpec);
    total_ac_p = sum(PowerSpec);
    
    if total_ac_p > 0
        purity = max_p / total_ac_p;
        % Threshold: 99% of energy in fundamental
        is_pure = purity > 0.9975; 
    else
        is_pure = false;
    end
end

fprintf('Identification Complete. Sinusoidal Samples: %d / %d\n', sum(Data.Sinusoidal), NumExp);

% Save to new file
[path, name, ext] = fileparts(FileName);
NewName = [name, '_Identified', ext];
save(NewName, 'Data');
fprintf('Saved updated dataset to: %s\n', NewName);
