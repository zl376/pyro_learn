import pyro
import torch
import numpy as np 
import pyro.distributions as dist

def linear_regression_model(x, y=None):
    # Define priors
    alpha = pyro.sample("alpha", dist.Normal(0., 10.))
    beta = pyro.sample("beta", dist.Normal(0., 10.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    # Define likelihood
    mu = alpha + beta * x
    with pyro.plate("data", x.shape[0]):
        return pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def logistic_regression_model_two_factors(x, y=None):
    # Define priors
    beta_r = pyro.sample("beta_r", dist.Normal(0.8, 0.1))
    # beta_r = pyro.sample("beta_r", dist.Uniform(0., 1.))
    beta_p = pyro.sample("beta_p", dist.Normal(0., 10.))

    # Define likelihood
    # pr = (1 / (1 + torch.exp(-beta_r))).expand_as(x)
    pr = beta_r.expand_as(x)
    pp = (1 / (1 + torch.exp(-beta_p))).expand_as(x)

    with pyro.plate("data", x.shape[0]):
        # Sample according to Bernoulli distribution parameterized with pr*pp
        return pyro.sample("obs", dist.Bernoulli(pr * pp), obs=y)

def generate_data(num_points=100, alpha_true=2.5, beta_true=0.9, sigma_true=0.5):
    x = torch.linspace(0, 10, num_points)
    y = alpha_true + beta_true * x + sigma_true * torch.randn(num_points)
    return x, y

def generate_data_two_factors(num_points=100, beta_r_true=0.5, beta_p_true=0.3):
    x = torch.linspace(0, 10, num_points)
    # pr = 1 / (1 + np.exp(-beta_r_true))
    pr = beta_r_true
    pp = 1 / (1 + np.exp(-beta_p_true))
    print(f'pr: {pr}, pp: {pp}, pr * pp: {pr * pp}')
    y = torch.bernoulli(torch.full_like(x, pr * pp) )
    return x, y

def generate_data_purchase(num_points = 100,
                           beta_price = -0.1,
                           beta_exposure = 0.3,
                           beta_intercept = 0,
                           beta_upload = 1):
    # features
    feat_price = np.random.uniform(20, 100, num_points) 
    feat_exposure = np.random.binomial(1, 0.2, num_points)
    x = np.stack([feat_price, feat_exposure], axis=1)
    # log odds 
    ## Add intercept column to x
    log_odds = np.dot(
        np.column_stack([x, np.ones(num_points)]), 
        np.array([beta_price, beta_exposure, beta_intercept])
    )
    log_likelihood = 1 / (1 + np.exp(-beta_upload)) * 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, log_likelihood)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def logistic_regression_model_purchase(x, y=None):
    # Define priors
    beta_price = pyro.sample("beta_price", dist.Normal(0, 10))
    beta_exposure = pyro.sample("beta_exposure", dist.Normal(0, 10))
    beta_intercept = pyro.sample("beta_intercept", dist.Normal(0, 10))
    loc_upload = -np.log(1 / 0.8 - 1)
    scale_upload = 0.05
    beta_upload = pyro.sample("beta_upload", dist.Normal(loc_upload, scale_upload))

    # Define likelihood
    # Create a tensor of beta coefficients
    beta = torch.stack([beta_price, beta_exposure, beta_intercept])
    
    # Augment x with an intercept column
    x_augmented = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    
    # Calculate log_odds using matrix multiplication
    log_odds = torch.matmul(x_augmented, beta)
    
    # Apply beta_upload
    probs = torch.sigmoid(beta_upload) * torch.sigmoid(log_odds)

    with pyro.plate("data", x.shape[0]):
        return pyro.sample("obs", dist.Bernoulli(probs), obs=y)
