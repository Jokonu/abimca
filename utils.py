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


def generate_synthetic_data(
    nr_of_samples: int = 400,
    column_names: list[str, str, str] = ["first", "second", "third"],
    moving_average_values=None,
    nr_of_instances: int = 12,
    seed: int = None,
) -> pd.DataFrame:
    nr_of_instances = 12
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
