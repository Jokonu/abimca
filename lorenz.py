import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def get_lorenz_attractor_dataframe(dt: float = 0.005, num_steps: int = 3000):
    parent_path = Path(__file__).parent.resolve()

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    data = np.array([xs, ys, zs]).T
    df = pd.DataFrame(data, columns=["xs", "ys", "zs"])
    return df


def plot_attractor(df: pd.DataFrame, filename="lorenz_attractor.png"):
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(df["xs"], df["ys"], df["zs"], lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz attractor")
    plt.savefig(filename)
    plt.close()


def main():
    df = get_lorenz_attractor_dataframe()
    plot_attractor(df)


if __name__ == "__main__":
    main()
