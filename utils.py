import pandas as pd
import numpy as np
import logging
from pathlib import Path
import coloredlogs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


GRAPHICS_FORMAT = "pdf"  # or png, pdf, svg, pgf
TRANSPARENT = False
RESOLUTION_DPI = 300


LOGPATH = Path.cwd() / "logs"
Path(LOGPATH).mkdir(parents=True, exist_ok=True)

def generate_random_continuous_synthetic_data(
    nr_of_samples: int = 400,
    column_names: list[str, str, str] = ["first", "second", "third"],
    moving_average_values=None,
    nr_of_instances: int = 2,
    seed: int = None,
    noise_factor: float = 0.02,
    n_classes: int = 10,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    dfs = []
    n_features = len(column_names)
    class_values = []
    class_label = []
    for cls in range(n_classes):
        # Initialize empty data matrix per class
        X = np.zeros((nr_of_samples, n_features), "float64")
        y = np.zeros((nr_of_samples, 1), "int")
        # Create random values between 0 and 1 for number of features
        x_hat = np.random.random(n_features)
        # Expand with gaussian noise to number of samples
        for feature in range(n_features):
            # Expand initial random value and put array in final instance matrix
            X[:, feature] = np.full(shape=nr_of_samples, fill_value=x_hat[feature])
            # Add noise with factor 'noise_factor'
            X[:, feature] = X[:, feature] + (np.random.random(size=nr_of_samples) * noise_factor)
            # Set label array
            y[:] = cls
        class_values.append(X)
        class_label.append(y)
    data = np.concatenate(class_values)
    label = np.concatenate(class_label)
    for instance in range(nr_of_instances):
        # Add random noise per instance
        data = data + (np.random.random(size=data.shape) * noise_factor)
        df = pd.DataFrame(data=data, columns=column_names)
        df["instance"] = instance
        df["class"] = label
        df.index.rename("time", inplace=True)
        df = df.set_index("instance", append=True)
        df = df.set_index("class", append=True)
        dfs.append(df)
    full_df = pd.concat(dfs)
    if moving_average_values is not None:
        full_df = full_df.rolling(moving_average_values, min_periods=1).mean()
    return full_df

def generate_synthetic_data(
    nr_of_samples: int = 400,
    column_names: list[str, str, str] = ["first", "second", "third"],
    moving_average_values=None,
    nr_of_instances: int = 12,
    seed: int = None,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    dfs = []
    for instance in range(nr_of_instances):
        nr_of_signals = 3
        h = nr_of_samples // 4
        x = np.zeros((nr_of_samples, nr_of_signals), "float64")
        y = np.zeros((nr_of_samples, 1), "float64")
        y[:, 0] = 0
        y[: h + 10, 0] = 1
        y[-2 * h :, 0] = 2
        y[-h:, 0] = 3
        aa = np.zeros((nr_of_samples), "float64")
        a = aa + 0.5 + 0.1 * np.random.random(nr_of_samples)
        x[:, 1] = a
        x[: h + 10, 1] = a[: h + 10] + 0.3
        x[-2 * h :, 1] = a[: 2 * h] + 0.3
        x[-h:, 1] = a[:h] - 0.3
        b = aa + 0.1 + 0.1 * np.random.random(nr_of_samples)
        x[:, 2] = b
        x[:h, 2] = b[:h] + 0.8
        x[-h:, 2] = b[:h] + 0.8
        c = aa + 0.2 + 0.1 * np.random.random(nr_of_samples)
        x[:, 0] = c
        x[h - 10 :, 0] = c[h - 10 :] + 0.5
        x[-h:, 0] = c[-h:]
        x[-2 * h :, 0] = c[-2 * h :]
        x = np.append(x, x, axis=0)
        y = np.append(y, y, axis=0)
        df = pd.DataFrame(data=x, columns=column_names)
        df["instance"] = instance
        df["class"] = y
        df.index.rename("time", inplace=True)
        df = df.set_index("instance", append=True)
        df = df.set_index("class", append=True)
        dfs.append(df)
    full_df = pd.concat(dfs)
    if moving_average_values is not None:
        full_df = full_df.rolling(moving_average_values, min_periods=1).mean()
    return full_df

def generate_lorenz_attractor_data(dt: float = 0.005, num_steps: int = 3000, nr_of_instances: int = 1):
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    def disfun(x, y, z):
        x0 = 2
        y0 = 10
        z0 = 23
        return (1 - (x / x0) + (y / y0)) * z0 - z

    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)

    # Set initial values
    xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)
    dfs = []
    for instance in range(nr_of_instances):
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps - 1):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s=(10 + instance * 0.5))
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
        data = np.array([xs, ys, zs]).T
        feature_names = ["xs", "ys", "zs"]
        x_label, y_label, z_label = feature_names
        time_index = np.arange(start=0, stop=num_steps * dt, step=dt)
        idx = pd.TimedeltaIndex(time_index, unit="S", name="time")
        df = pd.DataFrame(data, columns=feature_names, index=idx)
        label_array = np.zeros(df.shape[0])
        res = disfun(df[x_label], df[y_label], df[z_label])
        label_array = np.where(res < 1, 0, 1)
        df["label"] = label_array
        df["instance"] = instance
        df = df.set_index("label", append=True)
        df = df.set_index("instance", append=True)
        dfs.append(df)
    full_df = pd.concat(dfs)
    return full_df


def setupLogger(
    logging_file_path: Path = None,
    name: str = "jizzle",
    loglevel="INFO",
    log_to_file: bool = False,
):
    if loglevel is None:
        loglevel = "INFO"
    if logging_file_path is None:
        logging_file_name = LOGPATH / "logger.log"
        Path(LOGPATH).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    fmt = "%(asctime)s - %(name)s %(levelname)s %(funcName)s: %(message)s"
    if log_to_file is True:
        # create file handler which logs warning messages
        fh = logging.FileHandler(logging_file_name)
        fh.setLevel(loglevel)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    coloredlogs.install(level=loglevel, logger=logger, fmt=fmt)
    return logger
