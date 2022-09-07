# Standard Libraries
import os
from collections import defaultdict
from pathlib import Path

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

# Own Libraries
from abimca.subsequence_identifier import SubsequenceIdentifier

LOGGER_NAME = "abimca"
LOG_LEVEL = "DEBUG"
SUFFIX = "_lorenz"

TEMP_FOLDER = Path().cwd() / "abimca" / str(".abimca_tmp" + SUFFIX)
Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)

UNSUPERVISED_METRICS = {
    "silhouette_score": metrics.silhouette_score,
    "calinski_harabasz_score": metrics.calinski_harabasz_score,
    "davies_bouldin_score": metrics.davies_bouldin_score,
    # "mt3scm": mt3scm_score,
}


def prepare_training_data():
    df = utils.generate_lorenz_attractor_data(dt = 0.005, num_steps = 3000, nr_of_instances = 2)
    logger.debug(f"Scaling data..")
    data = df.copy()
    data.iloc[:, :] = StandardScaler().fit_transform(df.values)
    logger.debug(f"Scaling data DONE ..")
    data_train = data.xs(0, level="instance")
    data_test = data.xs(1, level="instance")
    data_test = data_train
    return data_train, data_test

def fit():
    logger.info("Running ABIMCA algorithm on the synthetic dataset.")
    data_train, _ = prepare_training_data()
    X_train = data_train.values
    logger.debug(f"{data_train=}")
    logger.info(f"Instantiating SubsequenceIdentifier class..")
    si = SubsequenceIdentifier(
        seq_len = 10,
        eta = 0.2,
        phi = 10,
        theta = 0.5,
        omega = 10,
        psi = 5,
        dim_bottleneck = 2,
        learning_rate = 1e-2,
        step_size = 1,
        disable_progress_bar=False,
        save_training_results=True,
        tempfolder_suffix=SUFFIX,
        set_parameters_automatically=False
    )
    logger.debug(f"{vars(si)=}")
    logger.info("")
    si.fit(X_train)
    logger.info("Saving temporary algorithm results for plotting..")
    dump(si, TEMP_FOLDER / "SubsequenceIdentifier_Lorenz.pkl")
    # params = si.get_params()
    logger.info("Done fitting.")


def predict(
    X_train=None,
    X_test=None,
    data_train=None,
    data_test=None,
    y_train=None,
    y_test=None,
):
    if X_train is None:
        data_train, data_test = prepare_training_data()
        X_train = data_train.values
        X_test = data_test.values

    si = load(TEMP_FOLDER / "SubsequenceIdentifier_Lorenz.pkl")
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
    plotting.plot_buffered_training_results(
        data_train,
        params["eta"],
        params["theta"],
        labels=y_train,
        signal_names=["xs", "ys", "zs"],
        # plot_signal_names=["x", "y", "z"],
        filename="online_clustering_lorenz",
        buffered_results_path=Path(TEMP_FOLDER / "buff_results.pkl"),
        buffered_reconstruction_path=Path(TEMP_FOLDER / "buff_reconstruction.pkl"),
        animation_plot=True,
    )
    plotting.plot_cluster_prediction(
        data_test=data_test,
        labels=labels_pred,
        x_label="xs",
        y_label="ys",
        z_label="zs",
        filename="offline_clustering_3d__lorenz.jpg",
    )
    plotting.plot_prediction(
        X=X_test,
        labels=labels_pred,
        ground_truth=labels_true,
        filename="offline_clustering_lorenz",
    )


if __name__ == "__main__":
    # Standard Libraries
    import logging

    utils.setupLogger(name=LOGGER_NAME, loglevel=LOG_LEVEL)
    logger = logging.getLogger(LOGGER_NAME)
    fit()
    predict()
