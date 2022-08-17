"""
Main plot script for generating graphics of the changepoint and subsequence identification comparisons

Author: Jonas Köhne
Date: 2022-01-20
License: See the LICENSE file.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
import utils
from mt3scm import MT3SCM
from lorenz import get_lorenz_attractor_dataframe

LOGGER_NAME = "abimca"
LOG_LEVEL = "DEBUG"
MAX_COLUMNS_TO_PLOT = 20
DEBUG_VALUES = 5000
DEVELOP = True
PLOTPATH = Path.cwd() / "plots" / datetime.today().strftime("%Y%m%d")
Path(PLOTPATH).mkdir(parents=True, exist_ok=True)

GRAPHICS_FORMAT = "pdf"  # or png, pdf, svg, pgf
TRANSPARENT = False
RESOLUTION_DPI = 300

def set_plot_params():
    # sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Computer Modern Roman"],
            "axes.grid": False,
            "image.cmap": cm.get_cmap("viridis")
        }
    )


def plot_buffered_training_results(
    orig_df,
    eta: float = 0.05,
    theta: float = 0.1,
    labels: np.array = None,
    signal_names: list[str] = None,
    filename: str = "trainbuffer",
    buffered_results_path: Path = Path("tmp") / "buff_results.pkl",
    buffered_reconstruction_path: Path = Path("tmp") / "buff_reconstruction.pkl",
):
    set_plot_params()
    new_sequence_loss_threshold = eta
    min_models_loss_for_sequence_recognition_threshold = theta
    alldata_df = pd.read_pickle(buffered_results_path)
    result_df = orig_df.copy()
    reconstruction_df = pd.read_pickle(buffered_reconstruction_path)
    result_df.loc[:] = reconstruction_df.values
    if ("anomaly" in result_df.index.names) and (
        "changepoint" in result_df.index.names
    ):
        extraplot = 1
    else:
        extraplot = 0
    df = alldata_df
    p1 = df[df.model == 0]
    nr_seqs = int(p1.subs_id.to_numpy().max())
    v = p1.subs_id.to_numpy()
    newidxs = np.where(v[:-1] != v[1:])[0]
    newidxs_stepsize = np.concatenate([[0], np.array(newidxs), [len(p1)]])
    newidxs = p1.iloc[newidxs].step.values
    newidxs = np.concatenate([[0], np.array(newidxs), [len(result_df)]]).astype("int")
    if signal_names is None:
        signames = orig_df.columns
    else:
        signames = signal_names
    sliced_orig_df = orig_df[signames]
    sliced_result_df = result_df[signames]
    figsize = [4, 6]
    figsize = set_size(250, subplots=(2, 1), invert_axis=False)
    fig, axs = plt.subplots(4 + extraplot, 1, figsize=figsize, constrained_layout=True)
    axs[0].plot(sliced_orig_df.to_numpy())
    for i in range(len(newidxs) - 1):
        axs[0].axvspan(
            newidxs[i],
            newidxs[i + 1],
            facecolor="C" + str(int(v[newidxs_stepsize[i] + 1])),
            alpha=0.3,
        )
    plot_sig_names = ["$x$", "$y$", "$z$"]
    axs[0].legend(plot_sig_names)
    axs[1].plot(sliced_result_df.to_numpy())
    recon_sig_names = [r"$\tilde{x}$", r"$\tilde{y}$", r"$\tilde{z}$"]
    axs[1].legend(recon_sig_names)
    axs[2].plot(p1["step"], p1["latent_mean"])
    axs[2].legend(["$h_b$"], loc="upper left")
    axs2twin = axs[2].twinx()
    axs2twin.plot(p1["step"], p1["subs_id"], color="orange")
    if labels is not None:
        axs2twin.plot(p1["step"], labels, color="green")
    axs2twin.legend(["$S_{ID}$", "ground truth"], loc="lower right")
    for s in range(0, nr_seqs + 1):
        p = df[df.model == s]
        axs[3].plot(p["step"], p["loss"])
    hline1 = axs[3].axhline(y=new_sequence_loss_threshold, color="k", label="$\eta$")
    hline2 = axs[3].axhline(
        y=min_models_loss_for_sequence_recognition_threshold,
        color="gray",
        label="$\zeta$",
    )
    axs[3].set_yscale("log")
    legs = [f"$s_{s}$" for s in range(1, nr_seqs + 1)]
    ax3_first_legend = axs[3].legend(["$s_b$"] + legs)
    axs[3].add_artist(ax3_first_legend)
    axs[3].legend(handles=[hline1, hline2], loc="lower right")
    axs[3].set_xlabel("time steps")
    if ("anomaly" in orig_df.index.names) and ("changepoint" in orig_df.index.names):
        anomalies = result_df.index.get_level_values("anomaly")
        changepoints = result_df.index.get_level_values("changepoint")
        axs[4].plot(p1["step"], anomalies)
        axs[4].plot(p1["step"], changepoints)
        axs[4].legend(["anomalies", "changepoints"], loc="best")
    plot_name = str(filename + "." + GRAPHICS_FORMAT)
    plt.savefig(
        PLOTPATH / plot_name,
        pad_inches=0,
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
        bbox_inches="tight",
    )
    plt.close()

def plot_cluster_prediction(
    data_test: pd.DataFrame,
    labels: np.array,
    x_label: str = None,
    y_label: str = None,
    z_label: str = None,
    filename: str = "SeqClustering-ClusterPrediction.jpg",
    plot_changepoints=False,
):
    if z_label is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        if len(np.unique(labels)) < 3:
            cmap = plt.cm.Set1
        else:
            cmap = plt.cm.tab20
        scatter = ax.scatter(
            data_test[x_label],
            data_test[y_label],
            data_test[z_label],
            s=0.5,
            cmap=cmap,
            c=labels,
        )
        if plot_changepoints is True:
            if len(np.unique(labels)) > 2:
                labs = np.where(labels[:-1] != labels[1:], 1, 0)
                labs = np.concatenate([[0], labs])
                data_changepoints = data_test[labs == 1]
            else:
                data_changepoints = data_test[labels == 1]
            ax.scatter(
                data_changepoints[x_label],
                data_changepoints[y_label],
                data_changepoints[z_label],
                s=20,
                c="r",
                marker="o",
            )
        else:
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(
                *scatter.legend_elements(), loc="lower left", title="Classes"
            )
            ax.add_artist(legend1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.tight_layout()
        plt.savefig(PLOTPATH / filename, pad_inches=0, bbox_inches="tight")
        plt.close()
    else:
        plt.scatter(
            data_test[x_label],
            data_test[y_label],
            s=labels * 8,
            cmap=plt.cm.tab20,
            c=labels,
        )
        plt.scatter(data_test[x_label], data_test[y_label], s=2, c="grey")
        plt.savefig(PLOTPATH / filename)
        plt.close()



def plot_prediction(
    X: np.array,
    labels: np.array,
    # instances: np.array = None,
    # eta: float = 0.05,
    # theta: float = 0.1,
    ground_truth=None,
    anomalies_predicted=None,
    filename: str = "SeqClusteringPrediction",
):
    set_plot_params()
    try:
        buffered_prediction_losses_path = (
            "mt3sid/.mt3sid_tmp/buff_prediction_results.pkl"
        )
        prediction_losses_df = pd.read_pickle(buffered_prediction_losses_path)
    except:
        prediction_losses_df = None
    steps = np.arange(X.shape[0])
    newidxs = np.where(labels[:-1] != labels[1:])[0]
    newidxs = np.concatenate([[0], np.array(newidxs), [X.shape[0]]])
    if ground_truth is not None:
        extraplot = 1
    else:
        extraplot = 0
    figsize = set_size(250, subplots=(1, 1), invert_axis=False)
    figsize = (3.46, 3.46)
    if prediction_losses_df is not None:
        fig, axs = plt.subplots(
            3 + extraplot, 1, figsize=figsize, constrained_layout=True
        )
    else:
        fig, axs = plt.subplots(
            2 + extraplot, 1, figsize=figsize, constrained_layout=True
        )
    # axs[0].set_title("Input data $X$")
    plot_sig_names = ["$x$", "$y$", "$z$"]
    axs[0].plot(X)
    axs[0].legend(plot_sig_names)
    for i in range(len(newidxs) - 1):
        axs[0].axvspan(
            newidxs[i],
            newidxs[i + 1],
            facecolor="C" + str(int(labels[newidxs[i] + 1])),
            alpha=0.3,
        )
    if prediction_losses_df is not None:
        # axs[2].set_title("Scores")
        axs[2].plot(prediction_losses_df)
        hline1 = axs[2].axhline(
            y=prediction_losses_df.attrs["eta"], color="k", label="$\eta$"
        )
        hline2 = axs[2].axhline(
            y=prediction_losses_df.attrs["theta"], color="gray", label="$\zeta$"
        )
        # axs[2].legend(prediction_losses_df.columns.to_list())
        axs[2].set_yscale("log")
        # axs[2].set_ylim(bottom=1e-2, top=1e0)
        nr_seqs = len(prediction_losses_df.columns.to_list())
        legs = [f"$s_{{{s}}}$" for s in range(1, nr_seqs + 1)]
        ax3_first_legend = axs[2].legend(legs, loc="lower left")
        axs[2].add_artist(ax3_first_legend)
        axs[2].legend(handles=[hline1, hline2], loc="right")
        # axs[2].legend(["$s_b$"] + legs)
    # axs[1].set_title("Subsequence ID $S_{ID}$")
    axs[1].plot(steps, labels, color="orange")
    axs[1].legend(["$S_{ID}$"])
    if (ground_truth is not None) and (anomalies_predicted is not None):
        # axs[3].set_title("Anomalies")
        axs[3].plot(steps, ground_truth)
        axs[3].plot(steps, anomalies_predicted)
        axs[3].legend(["ground_truth", "anomalies_predicted"])

    # plt.savefig(PLOTPATH / filename)
    plot_name = str(filename + "." + GRAPHICS_FORMAT)
    # plt.savefig(PLOTPATH / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
    plt.savefig(
        PLOTPATH / plot_name,
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT,
        pad_inches=0,
        bbox_inches="tight",
    )
    plt.close()



def set_size(width_pt, fraction=1, subplots=(1, 1), invert_axis: bool = False):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    if invert_axis is True:
        return (fig_height_in, fig_width_in)
    else:
        return (fig_width_in, fig_height_in)

def ax_scatter_3d_original(X, Y, Z, ax, kappa: np.ndarray, tau: np.ndarray, xyz_labels: list[str], subplot_title: str = "Subplot Title", marker_size: float = 0.8, marker_size_array=None, marker="o", plot_changepoints: bool = True, alpha=0.3):
    cmap = cm.get_cmap("viridis")
    marker="."
    data = kappa
    data = (data - data.min()) / (np.std(data))
    color = np.log((np.abs(data* 10) + 1) ** 2)*1
    scat = ax.scatter(X, Y, Z, c=color, cmap=cmap, s=color, marker=marker)
    ax.set_title(subplot_title)
    ax.set_xlabel(f"{xyz_labels[0]} [-]")
    ax.set_ylabel(f"{xyz_labels[1]} [-]")
    ax.set_zlabel(f"{xyz_labels[2]} [-]")
    ax.tick_params(labelsize=8)
    fig = plt.gcf()
    clb = fig.colorbar(scat, ax=ax, shrink=0.5, pad=0.2)
    clb.set_ticks([color.min(),color.max() / 2, color.max()])
    clb.set_ticklabels(['Low', 'Medium', 'High'], rotation = 45)
    clb.ax.tick_params(labelsize=8)


def create_3d_figure(df: pd.DataFrame, xyz_labels: list[str] = ["x", "y", "z"], predicted_class_array: np.ndarray = None, plot_changepoints: bool = True):
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    X = df.values
    label_array = np.ones(X.shape[0])
    label_array[0] = 0
    mt3 = MT3SCM()
    if predicted_class_array is None:
        predicted_class_array = np.ones(X.shape[0])
        predicted_class_array[0] = 0
    mt3scm_metric = mt3.mt3scm_score(X, predicted_class_array, standardize_subs_curve=True)
    logger.debug(f"{mt3scm_metric=}")
    kappa = np.log((np.abs(mt3.kappa_X * 1) + 1) ** 1)
    # tau = np.log((np.abs(mt3.tau_X * 10) + 1) ** 2)
    tau_X = mt3.tau_X
    ax_scatter_3d_original(X[:, 0], X[:, 1], X[:, 2], ax, kappa=kappa, tau=tau_X, xyz_labels=xyz_labels, subplot_title=None)
    return fig


def plot_lorenz_attractor_3d():
    df = get_lorenz_attractor_dataframe()
    set_plot_params()
    fig = create_3d_figure(df)
    plot_name = str("lorenz-attractor-3d." + GRAPHICS_FORMAT)
    fig.savefig(PLOTPATH / plot_name, pad_inches=0, bbox_inches="tight", transparent=TRANSPARENT, dpi=RESOLUTION_DPI, format=GRAPHICS_FORMAT)
    plt.close(fig)
    logger.debug(f"Plotted file {plot_name} to {PLOTPATH} successfully.")


if __name__ == "__main__":
    # Standard Libraries
    import logging

    utils.setupLogger(name=LOGGER_NAME, loglevel=LOG_LEVEL)
    logger = logging.getLogger(LOGGER_NAME)
    plot_lorenz_attractor_3d()
