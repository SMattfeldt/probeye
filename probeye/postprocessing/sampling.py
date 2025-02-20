# standard library
from typing import Union, Optional, TYPE_CHECKING

# third party imports
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import add_index_to_tex_prm_name

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inference_problem import InferenceProblem


def create_pair_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InferenceProblem",
    plot_with: str = "arviz",
    plot_priors: bool = True,
    focus_on_posterior: bool = False,
    kind: str = "kde",
    figsize: Optional[tuple] = (9, 9),
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    true_values: Optional[dict] = None,
    show_legends: bool = True,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a pair-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inference problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    plot_priors
        If True, the prior-distributions are included in the marginal subplots.
        Otherwise the priors are not shown.
    focus_on_posterior
        If True, the marginal plots will focus on the posteriors, i.e., the range of the
        horizontal axis will adapt to the posterior. This might result in just seeing a
        fraction of the prior distribution (if they are included). If False, the
        marginal plots will focus on the priors, which will have a broader x-range. If
        plot_priors=False, this argument has no effect on the generated plot.
    kind
        Type of plot to display ('scatter', 'kde' and/or 'hexbin').
    figsize
        Defines the size of the generated plot in the default unit. If None is chosen,
        the figsize will be derived automatically.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    true_values
        Used for plotting 'true' parameter values. Keys are the parameter names and
        values are the values that are supposed to be shown in the marginal plots.
    show_legends
        If True, legends are shown in the marginal plots. Otherwise no legends are
        included in the plot.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' pairplot function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
    """

    # a pairplot can only be generate when there are at least two parameter or parameter
    # components (the latter refers to vector-valued parameters)
    if problem.n_latent_prms_dim == 1:
        logger.warning(
            "The combined dimension of all latent parameters is one. Hence, no "
            "pairplot can be generated in this setup."
        )
        return np.array([])

    if plot_with == "arviz":

        # set default value for kde_kwargs if not given in kwargs; note that this
        # default value is mutable, so it should not be given as a default argument in
        # create_pair_plot
        if "kde_kwargs" not in kwargs:
            kwargs["kde_kwargs"] = {
                "contourf_kwargs": {"alpha": 0},
                "contour_kwargs": {"colors": None},
            }

        # process true_values if specified
        if true_values is not None:
            reference_values = dict()
            for prm_name, value in true_values.items():
                dim = problem.parameters[prm_name].dim
                tex = problem.parameters[prm_name].tex
                if dim > 1:
                    if tex in inference_data.posterior.keys():
                        # in this case, the inference_data object contains samples of a
                        # multidimensional parameter; the corresponding reference values
                        # must be given in a dict <tex>\n0, <tex>\n1, <tex>\n2, etc.
                        for i in range(dim):
                            key = f"{tex}\n{i}"
                            reference_values[key] = value[i]
                    else:
                        # in this case, the samples of a multidimensional parameter have
                        # been already decomposed, so all the channels in the inference
                        # data are 1D
                        for i in range(dim):
                            key = add_index_to_tex_prm_name(tex, i + 1)
                            reference_values[key] = value[i]
                else:
                    key = tex
                    reference_values[key] = value
            kwargs["reference_values"] = reference_values
            if "reference_values_kwargs" not in kwargs:
                kwargs["reference_values_kwargs"] = {"marker": "o", "color": "red"}

        # call the main plotting routine from arviz
        axs = az.plot_pair(
            inference_data,
            marginals=True,
            kind=kind,
            figsize=figsize,
            textsize=textsize,
            show=show,
            **kwargs,
        )

        # adds a reference value in each marginal plot; for some reason this is not done
        # by arviz.pair_plot when passing 'reference_values'
        if "reference_values" in kwargs:
            reference_values_kwargs = None
            if "reference_values_kwargs" in kwargs:
                reference_values_kwargs = kwargs["reference_values_kwargs"]
            ref_value_list = [*kwargs["reference_values"].values()]
            if problem.n_latent_prms_dim > 2:
                # in this case, the relevant axis is always the horizontal one
                for i, prm_value in enumerate(ref_value_list):
                    axs[i, i].scatter(
                        prm_value,
                        0,
                        label="true value",
                        zorder=10,
                        **reference_values_kwargs,
                        edgecolor="black",
                    )
            else:
                # in this case, the plot on the bottom right is rotated
                axs[0, 0].scatter(
                    ref_value_list[0],
                    0,
                    label="true value",
                    zorder=10,
                    **reference_values_kwargs,
                    edgecolor="black",
                )
                axs[1, 1].scatter(
                    0,
                    ref_value_list[1],
                    label="true value",
                    zorder=10,
                    **reference_values_kwargs,
                    edgecolor="black",
                )

        if plot_priors:

            # add the prior-pdfs to the marginal subplots
            prm_names = problem.get_theta_names(tex=False, components=False)
            i = 0  # not included in for-header due to possible dim-jumps
            for prm_name in prm_names:
                # for multivariate priors, no priors are plotted
                if problem.parameters[prm_name].dim > 1:
                    i += problem.parameters[prm_name].dim
                    continue
                x = None
                if focus_on_posterior:
                    if (problem.n_latent_prms_dim == 2) and (i == 1):
                        # the plot on the bottom right is rotated
                        x_min, x_max = axs[i, i].get_ylim()
                    else:
                        x_min, x_max = axs[i, i].get_xlim()
                    x = np.linspace(x_min, x_max, 200)
                # the following code adds labels to the prior and posterior plot if they
                # are represented as lines
                if axs[i, i].lines:
                    posterior_handle = [axs[i, i].lines[0]]
                    posterior_label = ["posterior"]
                else:
                    # this is for the case, when the posterior is not shown as a line,
                    # but for example as a histogram etc.
                    posterior_handle, posterior_label = [], []
                rotate = True if problem.n_latent_prms_dim == 2 and i == 1 else False
                problem.parameters[prm_name].prior.plot(
                    axs[i, i], problem.parameters, x=x, rotate=rotate
                )
                if show_legends:
                    prior_handle, prior_label = axs[i, i].get_legend_handles_labels()
                    axs[i, i].legend(
                        posterior_handle + prior_handle,
                        posterior_label + prior_label,
                        loc="best",
                    )
                i += 1

            # here, the axis of the non-marginal plots are adjusted to the new ranges
            if (not focus_on_posterior) and (problem.n_latent_prms_dim > 2):
                n = problem.n_latent_prms_dim
                for i in range(n):
                    x_min, x_max = axs[i, i].get_xlim()
                    for j in range(i + 1, n):
                        axs[j, i].set_xlim((x_min, x_max))
                    for j in range(0, i):
                        axs[i, j].set_ylim((x_min, x_max))
        else:

            # the following code adds legends to the marginal plots for the case where
            # no priors are supposed to be plotted
            if show_legends:
                prm_names = problem.get_theta_names(tex=False, components=True)
                for i, prm_name in enumerate(prm_names):
                    existing_handles, existing_labels = axs[
                        i, i
                    ].get_legend_handles_labels()
                    if axs[i, i].lines:
                        posterior_handle = [axs[i, i].lines[0]]
                        posterior_label = ["posterior"]
                    else:
                        # this is for the case, when the posterior is not shown as a
                        # line, but for example as a histogram etc.
                        posterior_handle, posterior_label = [], []
                    axs[i, i].legend(
                        posterior_handle + existing_handles,
                        posterior_label + existing_labels,
                        loc="best",
                    )

        # add a title to the plot, if requested
        if title:
            fig = plt.gcf()
            fig.suptitle(title, fontsize=title_size)

        # the following command reduces the otherwise wide margins; when only two
        # parameter (components) are given, the tight_layout()-call only results in a
        # warning without having an effect - hence, the if-clause
        if problem.n_latent_prms_dim > 2:
            plt.tight_layout()

        # by default, the y-axis of the first and last marginal plot have ticks, tick-
        # labels and axis-labels that are not meaningful to show on the y-axis; hence,
        # we remove them here; since the default plot looks different for only two
        # latent parameters, there is a check before
        if problem.n_latent_prms_dim > 2:
            for i in [0, -1]:
                axs[i, i].yaxis.set_ticks_position("none")
                axs[i, i].yaxis.set_ticklabels([])
                axs[i, i].yaxis.set_visible(False)
            for i in range(problem.n_latent_prms_dim - 1):
                axs[i, i].set_xticks(ticks=axs[-1, i].get_xticks())
                axs[i, i].set_xlim(axs[-1, i].get_xlim())
            axs[-1, -1].set_xticks(ticks=axs[-1, 0].get_yticks())
            axs[-1, -1].set_xlim(axs[-1, 0].get_ylim())

        return axs

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_posterior_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InferenceProblem",
    plot_with: str = "arviz",
    kind: str = "hist",
    figsize: Optional[tuple] = (10, 3),
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    hdi_prob: float = 0.95,
    true_values: Optional[dict] = None,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a posterior-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inference problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    kind
        Type of plot to display ('kde' or 'hist').
    figsize
        Defines the size of the generated plot in the default unit. If None is chosen,
        the figsize will be derived automatically.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    hdi_prob
        Defines the highest density interval. Must be a number between 0 and 1.
    true_values
        Used for plotting 'true' parameter values. Keys are the parameter names and
        values are the values that are supposed to be shown in the marginal plots.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_posterior function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
    """

    if plot_with == "arviz":

        # process true_values if specified
        if true_values is not None:
            var_names_raw = problem.get_theta_names(tex=False)
            ref_val = []
            for var_name in var_names_raw:
                if len_or_one(true_values[var_name]) == 1:
                    ref_val.append(true_values[var_name])
                else:
                    for true_value in true_values[var_name]:
                        ref_val.append(true_value)
            kwargs["ref_val"] = ref_val

        # call the main plotting routine from arviz and return the axes object
        axs = az.plot_posterior(
            inference_data,
            kind=kind,
            figsize=figsize,
            textsize=textsize,
            hdi_prob=hdi_prob,
            show=show,
            **kwargs,
        )

        # add a title to the plot, if requested
        if title:
            fig = plt.gcf()
            fig.suptitle(title, fontsize=title_size)

        return axs

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )


def create_trace_plot(
    inference_data: az.data.inference_data.InferenceData,
    problem: "InferenceProblem",
    plot_with: str = "arviz",
    kind: str = "trace",
    figsize: Optional[tuple] = (10, 6),
    textsize: Union[int, float] = 10,
    title_size: Union[int, float] = 14,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Creates a trace-plot for the given inference data.

    Parameters
    ----------
    inference_data
        Contains the results of the sampling procedure.
    problem
        The inference problem the inference data refers to.
    plot_with
        Defines the python package the plot will be generated with. Options are:
        {'arviz', 'seaborn', 'matplotlib'}.
    kind
        Allows to choose between plotting sampled values per iteration ("trace") and
        rank plots ("rank_bar", "rank_vlines").
    figsize
        Defines the size of the generated plot in the default unit. If None is chose,
        the figsize will be derived automatically.
    textsize
        Defines the font size in the default unit.
    title_size
        Defines the font size of the figures title if 'title' is given.
    title
        The title of the figure.
    show
        When True, the show-method is called after creating the plot. Otherwise, the
        show-method is not called. The latter is useful, when the plot should be further
        processed.
    kwargs
        Additional keyword arguments passed to arviz' plot_trace function.

    Returns
    -------
    axs
        The array of subplots of the created plot.
    """

    if plot_with == "arviz":

        # set default value for kde_kwargs if not given in kwargs; note that this
        # default value is mutable, so it should not be given as a default argument in
        # create_pair_plot
        if "plot_kwargs" not in kwargs:
            kwargs["plot_kwargs"] = {"textsize": textsize}

        # call the main plotting routine from arviz and return the axes object
        axs = az.plot_trace(
            inference_data, kind=kind, figsize=figsize, show=show, **kwargs
        )

        # add a title to the plot, if requested
        if title:
            fig = plt.gcf()
            fig.suptitle(title, fontsize=title_size)

        return axs

    elif plot_with == "seaborn":
        raise NotImplementedError(
            "The plot-creation with seaborn has not been implemented yet."
        )

    elif plot_with == "matplotlib":
        raise NotImplementedError(
            "The plot-creation with matplotlib has not been implemented yet."
        )

    else:
        raise RuntimeError(
            f"Invalid 'plot_with' argument: '{plot_with}'. Available options are "
            f"currently 'arviz', 'seaborn', 'matplotlib'"
        )
