"""
Simple linear regression example with two model parameters and one likelihood parameter
----------------------------------------------------------------------------------------
The model equation is y(x) = a * x + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is solved via max likelihood estimation and
via sampling using emcee, pyro and dynesty.
"""

# standard library imports
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_linear_regression(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        run_scipy: bool = True,
        run_emcee: bool = True,
        run_torch: bool = True,
        run_dynesty: bool = True,
    ):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps
            Number of steps (samples) to run. Note that the default number is rather low
            just so the test does not take too long.
        n_initial_steps
            Number of steps for initial (burn-in) sampling.
        n_walkers
            Number of walkers used by the estimator.
        plot
            If True, the data and the posterior distributions are plotted. This is
            deactivated by default, so that the test does not stop until the generated
            plots are closed.
        show_progress
            If True, progress-bars will be shown, if available.
        run_scipy
            If True, the problem is solved with scipy (maximum likelihood est).
            Otherwise, no maximum likelihood estimate is derived.
        run_emcee
            If True, the problem is solved with the emcee solver. Otherwise, the emcee
            solver will not be used.
        run_torch
            If True, the problem is solved with the pyro/torch solver. Otherwise, the
            pyro/torch solver will not be used.
        run_dynesty
            If True, the problem is solved with the dynesty solver. Otherwise, the
            dynesty solver will not be used.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of a, and its normal prior parameters
        a_true = 2.5
        loc_a = 2.0
        scale_a = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        loc_b = 1.0
        scale_b = 1.0

        # 'true' value of additive error sd, and its uniform prior parameters
        sigma = 0.5
        low_sigma = 0.1
        high_sigma = 0.8

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def definition(self):
                self.parameters = [{"a": "m"}, "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                # this method *must* be provided by the user
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

            def jacobian(self, inp: dict) -> dict:
                # this method *can* be provided by the user; if not provided the
                # jacobian will be approximated by finite differences
                x = inp["x"]  # vector
                one = np.ones((len(x), 1))
                # partial derivatives must only be stated for the model parameters;
                # all other input must be flagged by None; note: partial derivatives
                # must be given as column vectors
                return {"y": {"x": None, "m": x.reshape(-1, 1), "b": one}}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name; note that the name will
        # only be stored as an attribute of the InferenceProblem and is not important
        # for the problem itself; can be useful when dealing with multiple problems
        problem = InferenceProblem("Linear regression with normal additive error")

        # add all parameters to the problem; the first argument states the parameter's
        # global name (here: 'a', 'b' and 'sigma'); the second argument defines the
        # parameter type (three options: 'model' for parameter's of the forward model,
        # 'prior' for prior parameters and 'likelihood' for parameters of the likelihood
        # model); the 'info'-argument is a short description string used for logging,
        # and the tex-argument gives a tex-string of the parameter used for plotting;
        # finally, the prior-argument specifies the parameter's prior; note that this
        # definition of a prior will result in the initialization of constant parameters
        # of type 'prior' in the background
        problem.add_parameter(
            "a",
            "model",
            tex="$a$",
            info="Slope of the graph",
            prior=("normal", {"loc": loc_a, "scale": scale_a}),
        )
        problem.add_parameter(
            "b",
            "model",
            info="Intersection of graph with y-axis",
            tex="$b$",
            prior=("normal", {"loc": loc_b, "scale": scale_b}),
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            tex=r"$\sigma$",
            info="Standard deviation, of zero-mean additive model error",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )

        # add the forward model to the problem; note that the first positional argument
        # [{'a': 'm'}, 'b'] passed to LinearModel defines the forward model's parameters
        # by name via a list with elements structured like {<global parameter name>:
        # <local parameter name>}; a global name is a name introduced by problem.
        # add_parameter, while a local name is a name used in the response-method of the
        # forward model class (see the class LinearModel above); note that the use of
        # the local parameter name 'm' for the global parameter 'a' is added here only
        # to highlight the possibility of this feature; it is not necessary at all here;
        # whenever forward model's parameter has a similar local and global name (which
        # should be the case most of the times), one doesn't have to use the verbose
        # notation  {<global parameter name>: <local parameter name>} but can instead
        # just write the parameter's (global=local) name, like it is done with the
        # forward model's parameter 'b' below
        linear_model = LinearModel()
        problem.add_forward_model("LinearModel", linear_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal likelihood with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = linear_model.response(
            {linear_model.input_sensor.name: x_test, "m": a_true, "b": b_true}
        )[linear_model.output_sensor.name]
        y_test = np.random.normal(loc=y_true, scale=sigma)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={
                linear_model.input_sensor.name: x_test,
                linear_model.output_sensor.name: y_test,
            },
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel(linear_model.input_sensor.name)
            plt.ylabel(linear_model.output_sensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                              Add noise model(s)                              #
        # ============================================================================ #

        # add the noise model to the problem
        problem.add_likelihood_model(
            GaussianLikelihoodModel(prms_def={"sigma": "std_model"})
        )

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way; ref_values are used for plotting
        true_values = {"a": a_true, "b": b_true, "sigma": sigma}
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_torch=run_torch,
            run_dynesty=run_dynesty,
        )


if __name__ == "__main__":
    unittest.main()
