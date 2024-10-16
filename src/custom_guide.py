import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroParam
from pyro.distributions.constraints import positive, lower_cholesky

class MaskedPositive(torch.distributions.constraints.Constraint):
    def __init__(self, mask):
        self.mask = mask
        self.base_constraint = positive

    def check(self, value):
        return (self.mask * self.base_constraint.check(value) + 
                (1 - self.mask) * (value == 0))

class MaskedLowerCholesky(torch.distributions.constraints.Constraint):
    def __init__(self, mask):
        self.mask = mask
        self.base_constraint = lower_cholesky

    def check(self, value):
        return (self.mask * self.base_constraint.check(value) + 
                (1 - self.mask) * (value == 0))

class CustomStructuredGuide(AutoNormal):
    def __init__(self, model, variance_structure, init_loc_fn=None, init_scale=0.1):
        """
        Custom guide with structured variance for multivariate normal distribution.
        
        :param model: The model to be guided
        :param variance_structure: A dictionary specifying the structure of the variance
                                   for each parameter. The keys should be parameter names,
                                   and the values should be a tuple of (structure, mask),
                                   where structure is the desired variance structure
                                   (e.g., 'diagonal', 'full', or a custom torch.Tensor),
                                   and mask is a boolean tensor indicating which entries
                                   should be trainable (True) or fixed to zero (False).
        :param init_loc_fn: A per-site initialization function for loc parameters
        :param init_scale: Initial scale for the variance parameters
        """
        self.variance_structure = variance_structure
        super().__init__(model, init_loc_fn=init_loc_fn, init_scale=init_scale)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if site["type"] == "sample" and not site["is_observed"]:
                if name not in self.variance_structure:
                    raise ValueError(f"Variance structure not specified for parameter: {name}")
                
                structure, mask = self.variance_structure[name]
                
                if isinstance(structure, torch.Tensor):
                    init_scale = structure
                    constraint = MaskedPositive(mask)
                else:
                    raise ValueError(f"Unsupported variance structure for {name}: {structure}")
                
                scale = PyroParam(init_scale, constraint=constraint)
                self.scales.__setattr__(name, scale)

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if site["type"] == "sample" and not site["is_observed"]:
                loc = self.locs.__getattr__(name)
                scale = self.scales.__getattr__(name)
                
                with plates:
                    result[name] = pyro.sample(name, dist.MultivariateNormal(loc, scale_tril=scale))

        return result
