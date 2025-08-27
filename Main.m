% Calibrate Heston (via FFT pricer) to market IV surface,
% then compute and display standard errors and final loss.

clear all; clc; close all;

%% Settings
settings = calibrationSettings;  

%% Input data (volatility surfaces)
load('empVolatilitySurfaceData.mat'); % data

%% Initial parameters
parameters0 = settings.parameters0; % [kappa, theta, eta, rho, V0]

%% Calibrate model
settings.standardErrors = false; % turn off SEs inside pricingError
g = @(parameters)pricingError(data, settings, parameters);
fInitial= g(parameters0);

[parametersFinal, eFinal, exitFlag] = ...
    fminsearch(g, parameters0, settings.calibrOptions);

kappaF = parametersFinal(1);
thetaF = parametersFinal(2);
etaF   = parametersFinal(3);
rhoF   = parametersFinal(4);
V0F    = parametersFinal(5);

%% Compute standard errors for parameter estimates
n = size(data.IVolSurf,1)*size(data.IVolSurf,2); % The number of option contracts
f = g(parametersFinal)*n;

settings.standardErrors = true;           % turn on SEs path inside pricingError
fis = @(x)pricingError(data, settings, x);

% degrees of freedom in the problem (number of parameters)
p = length(parameters0);         % # of parameters (should be 5)

% estimate Jacobian
J = jacobianest(fis, parametersFinal);

% Covariance matrix
sigma2 = f / (n - p);
Sigma = sigma2*inv(J'*J);

% Parameter standard errors
se = sqrt(diag(Sigma))';

%% Display results
disp(['Estimated values: ', num2str(parametersFinal')]);
disp(['t -values: ', num2str(parametersFinal'./se)]);
disp(['Standard errors: ', num2str(se)]);
disp(['In-sample MSE: ', num2str(eFinal)]);

%% Price the Asian call with Heston

%% Load calibration results and market data
load('empVolatilitySurfaceData.mat');   % provides data.S0, data.r, data.K, data.T, data.IVolSurf
%parametersFinal = [0.011451, 0.024387, 3.2449, -0.64888, 0.054486]; % from previous calibration

% Unpack calibrated parameters under Q
kappa = parametersFinal(1);
theta = parametersFinal(2);
eta   = parametersFinal(3);
rho   = parametersFinal(4);
V0    = parametersFinal(5);

%% Simulation settings
S0    = data.S0;
r     = data.r;
H     = 0.85;          % Down-and-in barrier
T     = 1;             % maturity
M     = 100000;        % number of paths
dt    = 1/252;         % daily steps
N     = round(T/dt);   % time-steps per path
rng(12345);            % reproducibility

%% Simulate paths with antithetic variates
% Preallocate
p      = zeros(M,1);
p_anti = zeros(M,1);
S_store      = zeros(M, N);   % pre-allocate storage for the “current” path at each t
S_anti_store = zeros(M, N);

% Generate standardized normals
Z1 = randn(M,N);
Z2 = rho*Z1 + sqrt(1-rho^2)*randn(M,N);
Z1_anti = -Z1;
Z2_anti = rho*Z1_anti + sqrt(1-rho^2)*(-randn(M,N));

% Initialize
S      = S0 * ones(M,1);
S_anti = S;
V      = V0 * ones(M,1);
V_anti = V;
hit    = false(M,1);
hit_anti = false(M,1);

for t = 1:N
    % Variance update (Milstein + truncation)
    V = max(V + kappa*(theta - V)*dt + eta*sqrt(V*dt).*Z2(:,t) + 0.25*eta^2*dt.*(Z2(:,t).^2 - 1), 0);
    V_anti = max(V_anti + kappa*(theta - V_anti)*dt + eta*sqrt(V_anti*dt).*Z2_anti(:,t) + 0.25*eta^2*dt.*(Z2_anti(:,t).^2 - 1), 0);
    % Stock price update
    S = S .* exp((r - 0.5*V)*dt + sqrt(V*dt).*Z1(:,t));
    S_anti = S_anti .* exp((r - 0.5*V_anti)*dt + sqrt(V_anti*dt).*Z1_anti(:,t));
    % Barrier check
    hit = hit | (S < H);
    hit_anti = hit_anti | (S_anti < H);
    % Store paths for averaging
    S_store(:,t)      = S;
    S_anti_store(:,t) = S_anti;
end

% Compute arithmetic averages
avgS      = mean(S_store,2);
avgS_anti = mean(S_anti_store,2);

% Payoff: down-and-in Asian call
p(hit) = max(S(hit) - avgS(hit),0);
p_anti(hit_anti) = max(S_anti(hit_anti) - avgS_anti(hit_anti),0);

%% Price under risk-free numeraire
disc = exp(-r*T);
cp = 0.5 * (p + p_anti) * disc;
price  = mean(cp);
%stderr = std(cp)/sqrt(M);

% Display
fprintf('Heston Asian barrier call price: %.6f\n', price);
%fprintf('Standard error: %.6f\n', stderr);

%% txt file

%fid = fopen('OptionPriceResults.txt','w');
%fprintf(fid, 'Heston model: Down-and-in Asian call option\n');
%fprintf(fid, '===========================================\n');
%fprintf(fid, 'Parameter estimates:\n');
%fprintf(fid, 'kappa = %.6f\n', parametersFinal(1));
%fprintf(fid, 'theta = %.6f\n', parametersFinal(2));
%fprintf(fid, 'eta   = %.6f\n', parametersFinal(3));
%fprintf(fid, 'rho   = %.6f\n', parametersFinal(4));
%fprintf(fid, 'V0    = %.6f\n\n', parametersFinal(5));
%fprintf(fid, 'Option price  = %.6f\n', price);
%fprintf(fid, 'MSE = %.6f\n', eFinal);
%fprintf(fid, 'Std. error: \n');
%fprintf(fid, 'kappa = %.6f\n', se(1));
%fprintf(fid, 'theta = %.6f\n', se(2));
%fprintf(fid, 'eta = %.6f\n', se(3));
%fprintf(fid, 'rho = %.6f\n', se(4));
%fprintf(fid, 'V0 = %.6f\n', se(5));
%fclose(fid);



