import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spirals(n, cycles=2, sd=0):
    # n: number of points
    # sd: noise level

    np.random.seed()

    def one_spiral(n, cycles=1, sd=0):
        w = np.arange(start=0, stop=cycles, step=cycles / n)
        x = np.zeros((n, 2), dtype=float)
        x[:, 0] = (2 * w + 1) * np.cos(2 * np.pi * w) / 3
        x[:, 1] = (2 * w + 1) * np.sin(2 * np.pi * w) / 3
        if sd > 0:
            e = np.random.normal(scale=sd, size=n)
            xs = np.cos(2 * np.pi * w) - np.pi * (2 * w + 1) * np.sin(2 * np.pi * w)
            ys = np.sin(2 * np.pi * w) + np.pi * (2 * w + 1) * np.cos(2 * np.pi * w)
            nrm = np.sqrt(xs ** 2 + ys ** 2)
            x[:, 0] += e * ys / nrm
            x[:, 1] += e * xs / nrm
        return x

    x = np.zeros((n, 2), dtype=float)
    cl = np.zeros((n, 1), dtype=int)
    index_1 = np.random.choice(np.arange(n, dtype=int), size=int(n / 2), replace=False).tolist()
    index_2 = [i for i in np.arange(n, dtype=int) if i not in index_1]
    cl[index_2, 0] = 1
    x[index_1, :] = one_spiral(n=len(index_1), cycles=cycles, sd=sd)
    x[index_2, :] = -one_spiral(n=len(index_2), cycles=cycles, sd=sd)
    df = pd.DataFrame(np.concatenate((x, cl), axis=1), columns=["x", "y", "class"])
    df["class"] = df["class"].astype(int)
    plt.scatter(df["x"], df["y"], s=5, c=df["class"])
    plt.show()
    df.to_csv("spirals.csv")  # save dataset to "spirals.csv" in current directory
    return df


def blobs(n, center, cov):
    # n: number of points from each cluster
    # center: a list of mean of the N-dimensional distribution
    # cov: a list of N x N covariance matrix of the distribution
    holder = []
    np.random.seed()
    for cl, (mean, sigma) in enumerate(zip(center, cov)):
        holder.append(np.concatenate((np.random.multivariate_normal(mean, sigma, n), np.ones((n, 1), dtype=int) * cl), axis=1))
    df = pd.DataFrame(np.random.permutation(np.concatenate(tuple(holder), axis=0)), columns=list(range(center[0].shape[0])) + ["class"])
    df["class"] = df["class"].astype(int)
    df.to_csv("blobs.csv")
    return df


if __name__ == "__main__":
    data = spirals(n=1000, cycles=2, sd=0.05)
    data = blobs(200, [np.array([1, 2]), np.array([5, 6])], [np.array([[0.25, 0], [0, 0.25]])] * 2)
    # plt.scatter(data[0], data[1], s=5, c=data["class"])
    # plt.show()


