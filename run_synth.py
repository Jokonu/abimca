# Standard Libraries
import os
from collections import defaultdict
from pathlib import Path
import time

# Third Party Libraries
import utils
import plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import signal
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import interpolate

# Own Libraries
from abimca.subsequence_identifier import SubsequenceIdentifier

LOGGER_NAME = "abimca"
LOG_LEVEL = "DEBUG"
SUFFIX = "_synth"

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
    # n_classes = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    # n_classes = [1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500]
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
                    eta = 0.01,
                    phi = 10,
                    theta = 0.02,
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
                data = {"time": t_trainloop, "n_cls_found": n_uniq}
                index = {"n_cls": n_cls, "n_inst": n_inst, "n_samples": n_samples}
                multiindex = pd.MultiIndex.from_tuples([tuple(index.values())], names=index.keys())
                dfs.append(pd.DataFrame(data=data, index=multiindex))
    times = np.asarray(times)
    n_classes = np.asarray(n_classes)
    n_classes_found = np.asarray(n_classes_found)
    n_datapoints = np.asarray(n_datapoints)
    import pdb;pdb.set_trace()
    df_full = pd.concat(dfs)
    df = pd.DataFrame(data=np.array((times, n_classes, n_classes_found, n_datapoints)).T, columns=["time", "n_cls", "n_cls_found", "n_datapoints"])
    import pdb;pdb.set_trace()
    logger.info(f"{df}")
    df.to_pickle("complexity_results_v01.pkl")

def plot_results(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_pickle("complexity_results_v01.pkl")
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

    import seaborn as sns
    from plotting import set_plot_params
    set_plot_params()
    sns.regplot(data=df, x="n_cls_found", y="time", order=2)
    # sns.lmplot(data=df, x="n_cls_found", y="time", order=2)
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


    xs = df["n_cls_found"].values.astype("int")
    ys = df["n_datapoints"].values.astype("int")
    zs = df["time"].values
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    ax.legend()
    ax.set_xlabel("nr. of subsequences identified [-]")
    ax.set_ylabel("nr. of total datapoints scanned [-]")
    ax.set_zlabel("duration [s]")
    fig.savefig("complexity3d")
    plt.tight_layout()
    plt.close(fig)


if __name__ == "__main__":
    # Standard Libraries
    import logging

    utils.setupLogger(name=LOGGER_NAME, loglevel=LOG_LEVEL)
    logger = logging.getLogger(LOGGER_NAME)
    estimate_complexity()
    plot_results()
    # fit()
    # predict()
