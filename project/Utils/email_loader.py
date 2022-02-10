import pandas as pd
import numpy as np
from Utils.Sample import Sample


def load_email_data():
    data = pd.read_csv('./project/Datasets/spam_ham/emails.csv')
    data = data.to_numpy()
    np.random.shuffle(data)
    X = data[:, 1:5] # fix the indeces
    Y = data[:, -1]
    # TODO build Sample objects with label and a vector for X
    # Example:
    samples = list()
    for i in range(len(data)):
        sample = Sample(label=Y[i], X=X[i])
        samples.append(sample)
    return samples