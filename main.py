from abimca import SubsequenceIdentifier
import numpy as np


def main():
    # Generating random data. This will produce no class predictions or all points have the same class. For more reasonable results replace the data input with your mechatronic measurement data.
    # Number of datapoints (time-steps)
    n_p = 1000
    # Number of dimensions or features
    dim = 5
    X = np.random.rand(n_p, dim)
    # Number of clusters
    n_c = 5
    y = np.random.randint(n_c, size=n_p)

    # Compute online clustering
    si = SubsequenceIdentifier(disable_progress_bar=False)
    si.fit(X)
    print(f"Label array from online fitting: \n{si.label_array}")

    # Compute offline clustering
    labels = si.predict(X)
    print(f"Label array from online fitting: \n{labels}")


if __name__ == "__main__":
    main()