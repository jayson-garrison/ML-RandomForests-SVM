import pandas as pd
import numpy as np
from Utils.Sample import Sample
from Utils.Attribute import Attribute

def load_artificial_data(path):
    '''
    load the artificial data set given a path
    @param path
    @returns samples
    '''
    data = pd.read_csv(path)
    data = data.to_numpy()
    np.random.shuffle(data)
    X = data[1:, 1:-1] # i believe this is correct
    Y = data[1:, -1]
    # TODO build Sample objects with label and a vector for X
    # Example:
    samples = list()
    for i in range(len(Y)):
        sample = Sample(label=Y[i], X=X[i])
        samples.append(sample)
    return samples