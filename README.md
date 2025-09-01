## Heston Model Calibration & Exotic Option Pricing (MATLAB)

This repository implements the Heston stochastic volatility model end-to-end:  

(1) Calibrate parameters to a market implied volatility (IV) surface and  
(2) Price an exotic down-and-in Asian average-strike call by Monte Carlo under the calibrated model.

**Project Overview**

Calibration

FFT-based European call pricing under Heston for fast evaluation across strikes.

Nonlinear optimization via Nelder–Mead (fminsearch) minimizing IV-surface MSE.

Feasibility enforced via hard-constraint penalties in the objective.

Inference

Standard errors from numerical Jacobian and covariance 
Σ
=
𝜎
2
(
𝐽
⊤
𝐽
)
−
1
Σ=σ
2
(J
⊤
J)
−1
.

Exotic Pricing

Monte Carlo with Milstein (variance) + log-Euler (price), antithetic variates, barrier monitoring, and truncation to keep variance non-negative.

📂 Repo Structure
Main.m                          % Main script: calibration + standard errors + exotic pricing
pricingError.m                  % Calibration objective: IV(surface)_market vs IV(surface)_model (MSE or error vector)
CallPricingFFT.m                % FFT pricer for European calls under Heston
CharacteristicFunctionLib.m     % Characteristic functions (incl. Heston) used by FFT pricer
calibrationSettings.m           % Initial params, FFT grid, optimizer options
empVolatilitySurfaceData.mat    % Example data: {S0, r, K, T, IVolSurf}
jacobianest.m                   % Numerical Jacobian (for SEs) if not available on your path

What each key file does

Main.m – Runs everything:

calibrates Heston to the IV surface,

computes parameter SEs, t-values, MSE, and

prices the down-and-in Asian average-strike call via Monte Carlo.

pricingError.m – The objective function minimized during calibration.

Prices European calls with CallPricingFFT, converts to implied vols via blsimpv, compares to market IVs, and returns MSE (or the raw error vector when settings.standardErrors = true).

Applies large penalties when parameters violate 
𝜅
,
𝜃
,
𝜂
,
𝑉
0
>
0
κ,θ,η,V
0
	​

>0 or 
∣
𝜌
∣
>
1
∣ρ∣>1.

CallPricingFFT.m – FFT integrator that turns the Heston characteristic function into European call prices across strikes efficiently.

CharacteristicFunctionLib.m – Library of characteristic functions (Black–Scholes, Heston, Bates, etc.).

The Heston CF here is what CallPricingFFT uses to evaluate the pricing integral.

This file originates from Kienitz & Wetterau (2012); copyright header retained.

calibrationSettings.m – Sets the initial guess 
[
 
𝜅
,
𝜃
,
𝜂
,
𝜌
,
𝑉
0
 
]
[κ,θ,η,ρ,V
0
	​

], FFT grid size, and fminsearch options.

empVolatilitySurfaceData.mat – Example market inputs: spot 
𝑆
0
S
0
	​

, rate 
𝑟
r, strikes 
𝐾
K, maturities 
𝑇
T, and IV surface.

jacobianest.m – Finite-difference Jacobian utility used to compute parameter standard errors (if not provided by your environment).

▶️ How to Run

Clone the repo and open the folder in MATLAB.

Ensure the files above are on your MATLAB path.

Run:

run('Main.m')


Console output includes:

Estimated parameters, standard errors, t-values

In-sample MSE

Down-and-in Asian average-strike call price (Monte Carlo)

To export a text report, un-comment the “Export to .txt” block inside Main.m.

🧰 Methods (one-liners)

FFT pricing: invert Heston’s characteristic function to get call prices across strikes quickly.

IV comparison: convert model prices to implied vols (blsimpv) to compare with market IVs on a common scale.

Optimization: Nelder–Mead (fminsearch) minimizes IV-MSE; constraint violations penalized in the loss.

Monte Carlo: Milstein for variance, log-Euler for prices; antithetic variates, barrier flags, and positivity truncation.

Inference: Jacobian-based covariance for standard errors.

📊 Example Output (abridged)
Estimated values: 0.011451  0.024387  3.244900  -0.648880  0.054486
t -values:       3.16476    8.42829   12.72570  -7.81579    1.64093
Standard errors: 0.0036183  0.0028934  0.25499    0.083022   0.033204
In-sample MSE:   0.0010119

Heston Asian barrier call price: 3.XXXXXX


(values vary with data and RNG seed)
