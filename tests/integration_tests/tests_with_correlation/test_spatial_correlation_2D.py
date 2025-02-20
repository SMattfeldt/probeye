"""
Linear regression example with 2D spatial correlation model
----------------------------------------------------------------------------------------
The n data points (z1, z2, ..., zn) generated for this example are sampled from an
n-variate normal distribution with mean values given by zi = ax * xi + ay * yi + b with
a, b being the model parameters and x1, ..., xn and y1, ..., yn being predefined spatial
x- and y-coordinates ranging from 0 to 1. The data points (z1, z2, ..., zn) are not
independent but correlated in their spatial distance. The corresponding covariance
matrix is defined based on an exponential correlation function parameterized by the
constant standard deviation sigma of the n-variate normal distribution and a correlation
length l_corr. Hence, the full model has five parameters ax, ay, b, sigma, l_corr, all
of which are inferred in this example using maximum likelihood estimation as well as
sampling via emcee and dynesty.
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
    def test_spatial_correlation_2D(
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

        # 'true' value of ax, and its normal prior parameters
        ax_true = 1.5
        loc_ax = 2.0
        scale_ax = 1.0

        # 'true' value of ay, and its normal prior parameters
        ay_true = -1.0
        loc_ay = -0.5
        scale_ay = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        loc_b = 1.0
        scale_b = 1.0

        # 'true' value of additive error sd, and its uniform prior parameters
        sigma = 0.2
        low_sigma = 0.01
        high_sigma = 0.5

        # 'true' value of correlation length (x), and its uniform prior parameters
        l_corr = 0.1
        low_l_corr = 0.001
        high_l_corr = 0.2

        # settings for the data generation
        plot_data = False
        n_experiments = 1
        n_points = 10
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def definition(self):
                self.parameters = ["ax", "ay", "b"]
                self.input_sensors = [Sensor("x"), Sensor("y")]
                self.output_sensors = Sensor("z")

            def response(self, inp: dict) -> dict:
                ax = inp["ax"]
                ay = inp["ay"]
                b = inp["b"]
                x = inp["x"]
                y = inp["y"]
                return {"z": ax * x + ay * y + b}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Linear regression with normal additive error")

        # add all parameters to the problem
        problem.add_parameter(
            "ax",
            "model",
            tex="$a_x$",
            info="Slope of the graph-intersection with y=0 in x-direction",
            prior=("normal", {"loc": loc_ax, "scale": scale_ax}),
        )
        problem.add_parameter(
            "ay",
            "model",
            tex="$a_y$",
            info="Slope of the graph-intersection with x=0 in y-direction",
            prior=("normal", {"loc": loc_ay, "scale": scale_ay}),
        )
        problem.add_parameter(
            "b",
            "model",
            info="Intersection of plane with z-axis",
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

        # set the seed for the noise generation
        np.random.seed(seed)

        # first, create the true values without an error model; these 'true' values will
        # be the mean values for sampling from a multivariate normal distribution that
        # accounts for the intended correlation; the 'true' values are defined on an
        # xy-grid with the following discretization
        x_test_grid = np.linspace(0.0, 1.0, n_points)
        y_test_grid = np.linspace(0.0, 1.0, n_points)

        # now, we need to provide each point on the grid as an xy-pair; the x-values of
        # these pairs are stored in x_test, and the y-values in y_test
        x_test = np.zeros(n_points ** 2)
        y_test = np.zeros(n_points ** 2)
        i = 0
        for x_i in x_test_grid:
            for y_j in y_test_grid:
                x_test[i] = x_i
                y_test[i] = y_j
                i += 1

        # finally, compute the 'true' values on the grid using the forward model
        z_true = linear_model(
            {"ax": ax_true, "ay": ay_true, "b": b_true, "x": x_test, "y": y_test}
        )[linear_model.output_sensor.name]

        # assemble the spatial covariance matrix
        coords = np.zeros((n_points ** 2, 2))
        coords[:, 0] = x_test
        coords[:, 1] = y_test
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov = sigma ** 2 * correlation_matrix(coords, f_corr)

        # initialize a 3D-plot with the true mean-value-plane
        if plot_data:
            x_plot, y_plot = np.meshgrid(x_test_grid, y_test_grid)
            z_true_plot = z_true.reshape((n_points, -1))
            fig, axs = plt.subplots(subplot_kw={"projection": "3d"})
            axs.plot_wireframe(x_plot, y_plot, z_true_plot, label="true model")

        # now generate the noisy test data including correlations; we assume here that
        # there are n_experiments test series
        for i in range(n_experiments):
            exp_name = f"Test_{i}"
            z_test = np.random.multivariate_normal(mean=z_true, cov=cov)
            problem.add_experiment(
                exp_name,
                fwd_model_name="LinearModel",
                sensor_values={
                    linear_model.input_sensors[0].name: x_test,
                    linear_model.input_sensors[1].name: y_test,
                    linear_model.output_sensor.name: z_test,
                },
            )
            if plot_data:
                z_plot = z_test.reshape((n_points, -1))
                # noinspection PyUnboundLocalVariable
                axs.scatter(x_plot, y_plot, z_plot, label=f"measured data (test {i+1})")

        # show the plot
        if plot_data:
            axs.set_title("True model plus generated test data")
            axs.set_xlabel("x")
            axs.set_ylabel("y")
            axs.set_zlabel("z")
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
                correlation_variables="xy",
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
        true_values = {
            "ax": ax_true,
            "ay": ay_true,
            "b": b_true,
            "sigma": sigma,
            "l_corr": l_corr,
        }
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
