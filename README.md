# Heston Model Calibration & Exotic Option Pricing (MATLAB)

This repository implements the **Heston stochastic volatility model** end-to-end as required for the project in *Financial Mathematics and Statistics (DATA.STAT.610)*.

## Project Tasks
1. **Calibration of Heston parameters**  
   - Estimate Ψ = {κQ, θQ, η, ρ, V0} using a snapshot of the market implied volatility (IV) surface (`empVolatilitySurfaceData.mat`).  
   - The loss function minimised is the squared error between market IVs and model IVs across strikes and maturities.  
   - Optimisation is performed via **Nelder–Mead (`fminsearch`)**, with bounds specified in `calibrationSettings.m`.  
   - Pricing of European options under Heston is implemented using **FFT (`CallPricingFFT.m`)**.  
   - Outputs: estimated parameters, standard errors, and final loss function value (see `OptionPriceResults.txt`).  

2. **Exotic option pricing**  
   - Price a **down-and-in arithmetic Asian average-strike call** under the calibrated Heston model.  
   - Implemented using **Monte Carlo simulation** with:  
     - Daily time steps (Δt = 1/252), maturity T = 1  
     - Down barrier H = 0.85, 100,000 simulations  
     - **Variance reduction** with antithetic variates  
     - **Milstein scheme** with truncation to avoid negative variances  
   - The main entry point is `Main.m`, which ties together calibration and pricing.  

## Repository Contents
- `Main.m` → Runs calibration and exotic option pricing.  
- `CallPricingFFT.m` → FFT-based European option pricer under Heston.  
- `CharacteristicFunctionLib.m` → Characteristic function utilities.  
- `calibrationSettings.m` → Parameter bounds and optimisation settings.  
- `empVolatilitySurfaceData.mat` → Market implied volatility surface snapshot.  
- `pricingError.m`, `jacobianest.m` → Calibration utilities.  
- `OptionPriceResults.txt` → Results of parameter estimation and option pricing.  
- `README.md` → Project documentation (this file).  

## How to Run
1. Open MATLAB (R2023a or later recommended).  
2. Ensure all files are in the same working directory.  
3. Run:
   ```matlab
   run('Main.m')



(values vary with data and RNG seed)
