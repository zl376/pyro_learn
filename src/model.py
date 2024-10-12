import pyro
import torch
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

def generate_data(num_points=100, alpha_true=2.5, beta_true=0.9, sigma_true=0.5):
    x = torch.linspace(0, 10, num_points)
    y = alpha_true + beta_true * x + sigma_true * torch.randn(num_points)
    return x, y
