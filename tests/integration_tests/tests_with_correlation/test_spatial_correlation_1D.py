"""
Linear regression example with 1D spatial correlation model
----------------------------------------------------------------------------------------
The n data points (y1, y2, ..., yn) generated for this example are sampled from an
n-variate normal distribution with mean values given by yi = a * xi + b with a, b being
the model parameters and x1, x2, ..., xi, ..., xn being predefined spatial x-coordinates
ranging from 0 to 1. The data points (y1, y2, ..., yn) are not independent but
correlated in x. The corresponding covariance matrix is defined based on an exponential
correlation function parameterized by the constant standard deviation sigma of the
n-variate normal distribution and a correlation length l_corr. Hence, the full model has
four parameters a, b, sigma, l_corr, all of which are inferred in this example using
maximum likelihood estimation as well as sampling via emcee and dynesty.
"""

# standard library
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.utils import correlation_function
from tripy.utils import correlation_matrix

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_spatial_correlation_1D(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        run_scipy: bool = True,
        run_emcee: bool = True,
        run_torch: bool = False,
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

        if run_torch:
            raise RuntimeError(
                "The pyro-solver is not available for inference problems including "
                "correlations yet."
            )

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

        # 'true' value of correlation length, and its uniform prior parameters
        l_corr = 0.05
        low_l_corr = 0.001
        high_l_corr = 0.2

        # settings for the data generation
        plot_data = False
        n_experiments = 3
        n_points = 25
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def definition(self):
                self.parameters = ["a", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                a = inp["a"]
                b = inp["b"]
                x = inp["x"]
                return {"y": a * x + b}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Linear regression with normal additive error")

        # add all parameters to the problem
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
        problem.add_parameter(
            "l_corr",
            "likelihood",
            tex=r"$l_\mathrm{corr}$",
            info="Correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr, "high": high_l_corr}),
        )

        # add the forward model to the problem
        linear_model = LinearModel()
        problem.add_forward_model("LinearModel", linear_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; first create the true values without an error model; these
        # 'true' values will be the mean values for sampling from a multivariate normal
        # distribution that accounts for the intended correlation
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_points)
        y_true = linear_model({"a": a_true, "b": b_true, "x": x_test})[
            linear_model.output_sensor.name
        ]

        # assemble the spatial covariance matrix
        x_test_as_column_matrix = x_test.reshape((n_points, -1))
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov = sigma ** 2 * correlation_matrix(x_test_as_column_matrix, f_corr)

        # now generate the noisy test data including correlations; we assume here that
        # there are n_experiments test series
        for i in range(n_experiments):
            exp_name = f"Test_{i}"
            y_test = np.random.multivariate_normal(mean=y_true, cov=cov)
            problem.add_experiment(
                exp_name,
                fwd_model_name="LinearModel",
                sensor_values={
                    linear_model.input_sensor.name: x_test,
                    linear_model.output_sensor.name: y_test,
                },
            )
            if plot_data:
                plt.scatter(
                    x_test, y_test, label=f"measured data (test {i+1})", s=10, zorder=10
                )
        # finish the plot
        if plot_data:
            plt.plot(x_test, y_true, label="true model", c="black", linewidth=3)
            plt.xlabel("x")
            plt.ylabel(linear_model.output_sensor.name)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # since the different experiments are independent of each other they are put in
        # individual likelihood models (the problem's likelihood models are independent
        # of each other)
        for i in range(n_experiments):
            likelihood_model = GaussianLikelihoodModel(
                prms_def=[{"sigma": "std_model"}, "l_corr"],
                sensors=linear_model.output_sensor,
                correlation_variables="x",
                correlation_model="exp",
                experiment_names=f"Test_{i}",
                additive_model_error=True,
                multiplicative_model_error=False,
                additive_measurement_error=False,
            )
            problem.add_likelihood_model(likelihood_model)

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way
        true_values = {"a": a_true, "b": b_true, "sigma": sigma, "l_corr": l_corr}
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
