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
    X = data[1:, 1:-1] # i believe this is correct
    Y = data[1:, -1]
    samples = list()
    attr_dict = dict()
    for i in range(len(X)):
        if Y[i] == 0: Y[i] = -1 # NOTE: This change is necessary for SVM
        sample = Sample(label=Y[i], X=X[i])
        samples.append(sample)

        for j in range(len(X[i])):
            if j not in attr_dict:
                attr_dict[j] = set()
            attr_dict[j].add(X[i][j])
    
    attributes = list()
    for key in attr_dict:
        attributes.append(Attribute(name=key, values=attr_dict[key]))
        
    np.random.shuffle(samples)
    return samples, attributes