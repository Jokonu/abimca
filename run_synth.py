# Standard Libraries
import os
from collections import defaultdict
from pathlib import Path
import time

# Third Party Libraries
import utils
import plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import signal
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import seaborn as sns

# Own Libraries
from abimca.subsequence_identifier import SubsequenceIdentifier
from plotting import set_plot_params

LOGGER_NAME = "abimca"
LOG_LEVEL = "DEBUG"
SUFFIX = "_synth"
GRAPHICS_FORMAT = "png"  # or png, pdf, svg
RESOLUTION_DPI = 300
TRANSPARENT = False

TEMP_FOLDER = Path().cwd() / "abimca" / str(".abimca_tmp" + SUFFIX)
Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)

UNSUPERVISED_METRICS = {
    "silhouette_score": metrics.silhouette_score,
    "calinski_harabasz_score": metrics.calinski_harabasz_score,
    "davies_bouldin_score": metrics.davies_bouldin_score,
    # "mt3scm": mt3scm_score,
}


def prepare_training_data(nr_of_samples: int = 1000, nr_of_instances: int = 12):
    # df = utils.generate_synthetic_data(nr_of_samples=nr_of_samples, nr_of_instances=nr_of_instances, moving_average_values=20, seed = 42)
    df = utils.generate_random_continuous_synthetic_data(nr_of_samples=nr_of_samples, nr_of_instances=24, moving_average_values=20, seed=42)
    logger.debug(f"Scaling data..")
    data = df.copy()
    data.iloc[:, :] = StandardScaler().fit_transform(df.values)
    logger.debug(f"Scaling data DONE ..")
    data_test = data.xs(1, level="instance")
    data_train = data.xs(0, level="instance")
    return data_train, data_test


def fit(nr_of_samples: int = 1000, nr_of_instances: int = 12):
    logger.info("Running ABIMCA algorithm on the synthetic dataset.")
    data_train, _ = prepare_training_data()
    X_train = data_train.values
    logger.debug(f"{data_train=}")
    logger.info(f"Instantiating SubsequenceIdentifier class..")
    si = SubsequenceIdentifier(disable_progress_bar=False, save_training_results=True, tempfolder_suffix=SUFFIX)
    logger.debug(f"{vars(si)=}")
    logger.info("")
    si.fit(X_train)
    logger.info("Saving temporary algorithm results for plotting..")
    dump(si, TEMP_FOLDER / "SubsequenceIdentifier_Synth.pkl")
    # params = si.get_params()
    logger.info("Done fitting.")


def predict(nr_of_samples: int = 1000, nr_of_instances: int = 12):
    y_train = None
    if X_train is None:
        data_train, data_test = prepare_training_data()
        X_train = data_train.values
        X_test = data_test.values

    si = load(TEMP_FOLDER / "SubsequenceIdentifier_Synth.pkl")
    params = si.get_params()
    si.allow_unknown = False
    logger.debug(f"{params=}")
    labels = si.predict(X_test)

    labels_true = y_train
    labels_pred = labels

    if np.unique(labels_pred).shape[0] > 1:
        for name, metric in UNSUPERVISED_METRICS.items():
            result = metric(X_test, labels_pred)
            logger.debug(f"{name} = {result}")

    # metrics_dict, kappa_vec, tau_vec = calc_mt3scm(X_test, labels_pred)
    # logger.debug(f"{metrics_dict}")
    plotting.plot_cluster_prediction(
        data_test=data_test,
        labels=labels_pred,
        x_label="first",
        y_label="second",
        z_label="third",
        filename="offline_clustering_3d.jpg",
    )
    plotting.plot_buffered_training_results(
        data_train,
        params["eta"],
        params["theta"],
        labels=y_train,
        signal_names=["first", "second", "third"],
        # plot_signal_names=["x", "y", "z"],
        filename="online_clustering",
        buffered_results_path=Path(TEMP_FOLDER / "buff_results.pkl"),
        buffered_reconstruction_path=Path(TEMP_FOLDER / "buff_reconstruction.pkl"),
    )
    plotting.plot_prediction(
        X=X_test,
        labels=labels_pred,
        ground_truth=labels_true,
        filename="offline_clustering",
    )


def estimate_complexity():
    nr_of_samples = [10, 20, 30, 50, 100]
    nr_of_instances = [1, 2, 3, 4]
    moving_average_values = 3
    noise_factor = 0.01
    n_classes = [1, 2, 3, 5, 10, 20]
    column_names: list[str, str, str] = ["first", "second", "third"]
    # iterate over different number of classes and measure the time it takes to fit the data.
    # Check the actual number of subsequences found with the number of classes set for generating the data.
    times = []
    n_classes_found = []
    n_datapoints = []
    dfs = []
    for n_samples in nr_of_samples:
        for n_inst in nr_of_instances:
            for n_cls in n_classes:
                temp_folder_suffix = f"_complexity_{n_cls}_{n_inst}_{n_samples}"
                temp_folder = Path().cwd() / "abimca" / str(".abimca_tmp" + temp_folder_suffix)
                Path(temp_folder).mkdir(parents=True, exist_ok=True)
                df = utils.generate_random_continuous_synthetic_data(
                    nr_of_samples=n_samples,
                    nr_of_instances=n_inst,
                    moving_average_values=moving_average_values,
                    column_names=column_names,
                    seed=42,
                    n_classes=n_cls,
                    noise_factor=noise_factor,
                )
                n_datapoints.append(df.shape[0])
                si = SubsequenceIdentifier(
                    seq_len = 5,
                    eta = 0.005,
                    phi = 10,
                    theta = 0.0125,
                    omega = 10,
                    psi = 5,
                    dim_bottleneck = 2,
                    learning_rate = 5e-3,
                    step_size = 1,
                    disable_progress_bar=False,
                    save_training_results=True,
                    tempfolder_suffix=temp_folder_suffix,
                    set_parameters_automatically=False
                )
                start = time.time()
                si.fit(df.values)
                ende = time.time()
                t_trainloop = ende - start
                logger.info("Saving temporary algorithm results for plotting..")
                dump(si, temp_folder / f"abimca_complexity.pkl")
                y_train = df.index.get_level_values("class")
                params = si.get_params()
                n_uniq = np.unique(si.label_array).shape[0]
                n_classes_found.append(n_uniq)
                logger.debug(f"{n_uniq=}")
                plotting.plot_buffered_training_results(
                    df,
                    params["eta"],
                    params["theta"],
                    labels=y_train,
                    signal_names=column_names,
                    filename=f"online_clustering_{n_cls}_{n_inst}_{n_samples}",
                    buffered_results_path=Path(temp_folder / "buff_results.pkl"),
                    buffered_reconstruction_path=Path(temp_folder / "buff_reconstruction.pkl"),
                )
                plt.plot(df.values)
                plt.savefig(f"data_{n_cls}_{n_inst}_{n_samples}.png")
                plt.close()
                times.append(t_trainloop)
                logger.debug(f"time spend for {n_cls=}_{n_inst=}_{n_samples=}: {t_trainloop:.2f}")
                # data = {"time": t_trainloop, "n_cls": n_cls, "n_inst": n_inst, "n_samples": n_samples}
                data = {"time": t_trainloop, "n_cls_found": n_uniq, "n_datapoints": df.shape[0]}
                index = {"n_cls": n_cls, "n_inst": n_inst, "n_samples": n_samples}
                multiindex = pd.MultiIndex.from_tuples([tuple(index.values())], names=index.keys())
                dfs.append(pd.DataFrame(data=data, index=multiindex))
                df_full = pd.concat(dfs)
                df_full.to_pickle("complexity_results_full_v02.pkl")

def plot_results(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_pickle("complexity_results_full_v02.pkl")
    df = df.reset_index()
    plt.plot(df["n_cls_found"], df["time"])
    plt.xlabel("n_cls_found [-]")
    plt.ylabel("duration [s]")
    plt.title("Complexity estimation")
    plt.savefig("complexity_time_cls_found")
    plt.close()

    plt.plot(df["n_cls"], df["time"])
    plt.xlabel("n_cls [-]")
    plt.ylabel("duration [s]")
    plt.title("Complexity estimation")
    plt.savefig("complexity_time_cls")
    plt.close()

    plt.plot(df["n_datapoints"], df["time"])
    plt.xlabel("n_datapoints [-]")
    plt.ylabel("duration [s]")
    plt.title("Complexity estimation")
    plt.savefig("complexity_time_datapoints")
    plt.close()

    set_plot_params()
    sns.regplot(data=df, x="n_cls_found", y="time", order=2)
    sns.despine()
    plt.tight_layout()
    plt.savefig("complexity_sns")
    plt.close()

    set_plot_params()
    fig, ax = plt.subplots(layout="constrained")
    x = df["n_cls_found"].values.astype("int")
    y = df["time"].values
    poly_coefficients = np.polyfit(x=x, y=y, deg=2)
    polynom = np.poly1d(poly_coefficients)
    ax.scatter(x,y , label="Empirical results", c="C0")
    step_size = 1
    x_new = np.arange(x.min(), x.max() + step_size, step_size)
    y_new = polynom(x_new)
    ax.plot(x_new, y_new, label="Cubic regression line", c="C1")
    ax.legend()
    ax.set_xlabel("nr. of subsequences identified [-]")
    ax.set_ylabel("duration [s]")
    # ax.set_title("Complexity estimation")
    fig.savefig("complexity")
    plt.tight_layout()
    plt.close(fig)

def plot_curves(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_pickle("complexity_results_full_v02.pkl")
    df = df.reset_index()
    # find unique values for n_n_datapoints
    n_dps = df["n_datapoints"].unique()
    set_plot_params()
    fig, ax = plt.subplots(layout="constrained")
    for n_datapoints in n_dps:
        df_dp = df[df["n_datapoints"] == n_datapoints]
        x = df_dp["n_cls_found"].values.astype("int")
        y = df_dp["time"].values
        poly_coefficients = np.polyfit(x=x, y=y, deg=2)
        polynom = np.poly1d(poly_coefficients)
        ax.scatter(x,y , label="Empirical results", c="C0")
        step_size = 1
        x_new = np.arange(x.min(), x.max() + step_size, step_size)
        y_new = polynom(x_new)
        ax.plot(x_new, y_new, label=f"n_points = {n_datapoints}", c="C1")
    # ax.legend()
    ax.set_xlabel("nr. of subsequences identified [-]")
    ax.set_ylabel("duration [s]")
    # ax.set_title("Complexity estimation")
    plt.tight_layout()
    fig.savefig("complexity_curves")
    import pdb;pdb.set_trace()

    df2 = df.loc[(df["n_samples"] == 50) & (df["n_inst"] == 1)]
    plt.scatter(df2["n_datapoints"], df2["time"])
    plt.savefig("somemore_png")
    plt.close()

    sns.lmplot(data=df, x="n_datapoints", y="time", col="n_inst", row="n_samples", hue="n_inst", palette="muted", ci=None, height=4, scatter_kws={"s": 50, "alpha": 1}, order=2,  facet_kws={"sharex":False, "sharey":False})
    plt.savefig("somemore_png")


def plot_2d_results(df: pd.DataFrame = None):
    set_plot_params()
    if df is None:
        df = pd.read_pickle("complexity_results.pkl")
    df = df.reset_index()
    df = df.drop(11)
    df = df.drop(0)
    df = df.drop(1)
    x = df["n_datapoints"].values.astype("int")
    y = df["time"].values
    poly_coefficients = np.polyfit(x=x, y=y, deg=2)
    polynom = np.poly1d(poly_coefficients)
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(x,y , label="Empirical results", c="C0")
    step_size = 1
    x_new = np.arange(x.min(), x.max() + step_size, step_size)
    y_new = polynom(x_new)
    ax.plot(x_new, y_new, label=f"bla", c="C1")
    ax.set_xlabel("nr. of datapoints [-]")
    ax.set_ylabel("duration [s]")
    fig.savefig("complexity_2d_curve")
    plt.close()
    ax = sns.scatterplot(data=df, x="n_datapoints", y="time", size= "n_cls_found", sizes=(3, 300), alpha=1)
    # ax = sns.regplot(data=df, x="n_datapoints", y="time", scatter=True, order=2)
    # plt.xscale('log')
    # plt.yscale('log')
    ax.plot(x_new, y_new, label=f"2nd order polynomial regression", c="C1")
    # ax.legend()
    plt.tight_layout()
    plt.savefig("sns_complexity_2d_curve")
    plt.close()
    import pdb;pdb.set_trace()

def plot_3d_results(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_pickle("complexity_results_full_v02.pkl")
    df = df.reset_index()
    # set_plot_params()
    # Extract the data
    xs = df["n_cls_found"].values.astype("int")
    ys = df["n_datapoints"].values.astype("int")
    zs = df["time"].values

    x_data = xs
    y_data = ys
    z_data = zs
    from scipy.optimize import curve_fit
    # test function
    def function(data, a, b, c, d, e, f, g, h):
        x = data[0]
        y = data[1]
        # return a * (x**3) + b * x  + d * y + e
        # return a * (x**2) + b * (y**2) + c * x + d * y + e
        # return a * (x) + b * y + c * x*y + d
        return a * (x**2) + b * (y**2) + c * (x**2)*y + d * (y**2)*x + e * x * y + f * x + g * y + h
    # get fit parameters from scipy curve fit
    parameters, covariance = curve_fit(function, [x_data, y_data], z_data)

    # create surface function model
    mesh_points = 1000
    # setup data points for calculating surface model
    model_x_data = np.linspace(min(x_data), max(x_data), mesh_points)
    model_y_data = np.linspace(min(y_data), max(y_data), mesh_points)
    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(model_x_data, model_y_data)
    # calculate Z coordinate array
    Z = function(np.array([X, Y]), *parameters)
    # import pdb;pdb.set_trace()
    # Z = Z - np.tile(Z[:, 0], (mesh_points, 1)).T
    # setup figure object
    # reset data
    # fig = plt.figure(layout="constrained")
    # textwidth of two column paper: 17.75cm
    # fig = plt.figure(1, constrained_layout=False, figsize=(17.75*cm * 2, 17.75*cm / 1.5))
    n_x_subplots=1
    n_y_subplots=1
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8,
            "font.sans-serif": ["Computer Modern Roman"],
            "axes.grid": False,
            "image.cmap": cm.get_cmap("viridis")
        }
    )
    centimeter = 1/2.54
    # fig, ax = plt.subplots(constrained_layout=True, figsize=(17.75*centimeter*n_x_subplots/2, 17.75*centimeter*n_y_subplots/2), subplot_kw=dict(projection="3d"))
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    # setup 3d object
    # ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=25, azim=-50)
    # ax = Axes3D(fig)
    # X = np.log10(X)
    # Y = np.log10(Y)
    # Z = np.log10(Z)

    # x_data = np.log10(x_data)
    # y_data = np.log10(y_data)
    # z_data = np.log10(z_data)

    ax.scatter((x_data), (y_data), (z_data), color='red', label="empirical result")
    # plot surface
    surf = ax.plot_surface((X), (Y), (Z), lw=0.5, rstride=8, cstride=8, alpha=0.3, label="2nd order polynomial regression")
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax.contour(X, Y, Z, zdir='z', offset=(Z).min(), cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=(X).min(), cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=(Y).max(), cmap='coolwarm')
    # ax.plot_surface((X), (Y), Z)
    # plot input data
    # ax.scatter((x_data), (y_data), z_data, color='red')
    # ax.set_xscale("log")
    # ax.set_xscale("log")
    # ax.set_xscale("log")
    # set plot descriptions
    ax.legend()
    ax.set_xlabel('subsequences [-]')
    ax.set_ylabel('datapoints [-]')
    ax.set_zlabel('duration [s]')
    # plt.subplots_adjust(left=0)
    # plt.tight_layout()
    # fig.savefig("complexity3dfit")
    plot_name ="complexity3dfit"
    plt.savefig(
        str(plot_name + "." + GRAPHICS_FORMAT),
        pad_inches=0,
        bbox_inches="tight",
        transparent=TRANSPARENT,
        dpi=RESOLUTION_DPI,
        format=GRAPHICS_FORMAT
    )


if __name__ == "__main__":
    # Standard Libraries
    import logging

    utils.setupLogger(name=LOGGER_NAME, loglevel=LOG_LEVEL)
    logger = logging.getLogger(LOGGER_NAME)
    # estimate_complexity()
    # plot_results()
    plot_3d_results()
    # plot_2d_results()
    # plot_curves()
    # fit()
    # predict()
