#!/usr/bin/env python
# coding: utf-8
# ---atul.agrawal@tum.de----
# This just serves as an example to incorporate adjoint based differentiable PDE solver () in torch/pyro.
# In effect coupling Fenics and torch/pyro. This required fenics-adjoint package to work


from fenics import *
from fenics_adjoint import *

import torch

# need to install package, not of pypi so I dont know how to put in git
# https://github.com/barkm/torch-fenics
# The above package can be replaced by manual implementation also but why reinvent the wheel
import torch_fenics

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance, NUTS, MCMC, HMC
import seaborn as sns
import pandas as pd

torch.set_default_dtype(torch.float64)


class Poisson(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Create function space
        mesh = UnitIntervalMesh(20)
        self.V = FunctionSpace(mesh, "P", 1)

        # Create trial and test functions
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * dx

        # Construct boundary condition
        bc = DirichletBC(self.V, g, "on_boundary")

        # Solve the Poisson equation
        u = Function(self.V)
        solve(self.a == L, u, bc)

        # Return the solution
        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return Constant(0), Constant(0)


# # ---------------------
# ## Making it probabilistic
# Trying to infer f,g and the noise


# True Values

f = torch.tensor([[1.0]], dtype=torch.float64)
g = torch.tensor([[1.0]], dtype=torch.float64)
sigma = 0.05


u_true = poisson(f, g)


u_exp = np.random.normal(loc=u_true, scale=sigma)

plt.plot(u_exp[0, :], "*", u_true[0, :], label="exp1")


def pos_model():
    f = pyro.sample("f", dist.Uniform(0.8, 1.2))
    g = pyro.sample("g", dist.Uniform(0.8, 1.2))

    sigma_smpl = pyro.sample("sigma", dist.Uniform(0, 0.2))

    # solver inputs.Unclean but works
    poisson = Poisson()

    pyro.sample(
        "lkl",
        dist.Normal(
            poisson(f.unsqueeze(0).unsqueeze(0), g.unsqueeze(0).unsqueeze(0)),
            sigma_smpl,
        ),
        obs=th.from_numpy(u_exp),
    )


kernel = NUTS(pos_model)
mcmc_1 = MCMC(kernel, num_samples=300, warmup_steps=100)
mcmc_1.run()
mcmc_1.summary()


sns.pairplot(pd.DataFrame(mcmc_1.get_samples()), kind="kde")
