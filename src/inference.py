import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormal

def run_inference(model, x, y, num_iterations=1000):
    # Clear the param store in case we're rerunning this cell
    pyro.clear_param_store()

    # Setup the optimizer and the inference algorithm
    adam_params = {"lr": 0.03, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)
    svi = SVI(model, guide=AutoNormal(model), loss=Trace_ELBO(), optim=optimizer)

    # Do gradient steps
    for step in range(num_iterations):
        loss = svi.step(x, y)
        if step % 100 == 0:
            print(f"Elbo loss: {loss}")

    return svi
