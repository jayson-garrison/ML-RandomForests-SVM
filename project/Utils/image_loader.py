import pandas as pd
import numpy as np
from Utils.Sample import Sample
from Utils.Attribute import Attribute

def image_loader():
    data = pd.read_csv('./project/Datasets/mnist/train.csv') 
    data = data.to_numpy()
    np.random.shuffle(data)
    X = data[1:, 1:]
    Y = data[1:, 0]
    # TODO build Sample objects with label and a vector for X
    # Example:
    samples = list()
    for i in range(len(Y)):
        sample = Sample(label=Y[i], X=X[i])
        samples.append(sample)
    return samples