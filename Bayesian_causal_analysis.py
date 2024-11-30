import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

# Simulated Data
np.random.seed(42)
n = 500
treatment = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
covariate = np.random.normal(size=n)
sales = 50 + 10 * treatment + 5 * covariate + np.random.normal(size=n)

data = pd.DataFrame({"treatment": treatment, "covariate": covariate, "sales": sales})

# Bayesian Model
with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal("beta_0", mu=0, sigma=10)
    beta_T = pm.Normal("beta_T", mu=0, sigma=10)
    beta_X = pm.Normal("beta_X", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    # Expected Sales
    mu = beta_0 + beta_T * data["treatment"] + beta_X * data["covariate"]
    
    # Likelihood
    sales_obs = pm.Normal("sales_obs", mu=mu, sigma=sigma, observed=data["sales"])
    
    # Sampling
    trace = pm.sample(1000, return_inferencedata=True)

# Posterior Analysis
az.plot_posterior(trace, var_names=["beta_T"], hdi_prob=0.95)
