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
