function f = pricingError(data, settings, params)
% pricingError   Compute Heston‐FFT calibration loss or raw error vector,
%                with hard constraints on the parameters:
%                  kappa>0, theta>0, eta>0, V0>0, rho in [-1,1].
%
%   f = pricingError(data, settings, params)
%
%   If any constraint is violated, returns a huge penalty (or zero‐vector for SE mode).

  % unpack parameters
  kappa = params(1);
  theta = params(2);
  eta   = params(3);
  rho   = params(4);
  V0    = params(5);

  % Constraint check
  if kappa <= 0 || theta <= 0 || eta <= 0 || V0 <= 0 || abs(rho) > 1
    % infeasible → return large penalty (or zero‐vector for SE mode)
    if settings.standardErrors
      f = zeros(numel(data.T)*numel(data.K),1);
    else
      f = 1e10;
    end
    return
  end

  % Unpack market data
  T      = data.T(:);
  K      = data.K(:);
  iv_market = data.IVolSurf;
  S0     = data.S0;
  r      = data.r;
  [M, N] = size(iv_market);

  % Loop to compute per‐quote errors
  errors = zeros(M*N,1);
  idx = 1;
  for i = 1:M
    for j = 1:N
      % FFT‐price under Heston
      Cmodel = CallPricingFFT(settings.model, settings.n, ...
                   S0, K(j), T(i), r, 0, ...
                   kappa, theta, eta, rho, V0);
      % Invert to implied vol
      try
        iv_model = blsimpv(S0, K(j), r, T(i), Cmodel);
      catch
        iv_model = NaN;
      end
      if isnan(iv_model) || iv_model <= 0
        continue; % skip this point
      end
      % Record error
      errors(idx) = iv_market(i,j) - iv_model;
      idx = idx + 1;
    end
  end

  % Return either SSE or raw error vector
  if settings.standardErrors
    f = errors;          % vector for Jacobian/SE
  else
    f = mean(errors.^2);  % mean squared error
  end
end
