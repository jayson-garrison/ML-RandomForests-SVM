import pandas as pd
import numpy as np
from Utils.Sample import Sample
from Utils.Attribute import Attribute
from Utils.helper_functions import *
from Utils.PCA import pca

def load_image_data():
    # Restructure the data so that |X| is 255 (one feature for each intensity) and x_i is the count for the ith intensity in the image X_i
    data = pd.read_csv('./project/Datasets/mnist/train.csv')
    data = data.to_numpy()
    np.random.shuffle(data)
    indeces = set()
    while len(indeces) < 20000:
        indeces.add(np.random.randint(0, 42000))
    indeces = list(indeces)
    indeces.sort()

    # reading a pca file
    # X = data[1:, 2:]
    # Y = data[1:, 1]

    # reading original file
    X_o = data[1:, 1:]
    Y_o = data[1:, 0]

    # sampled
    X = np.take(X_o, indeces, axis=0)
    Y = np.take(Y_o, indeces, axis=0)

    # pca on orig?
    X = pca(X, 10)

    pca_emails = pd.DataFrame(np.column_stack((Y,X))) 
    pca_emails.to_csv('pca_image_20k_dim20.csv')

    samples = list()
    attr_dict = dict()
    for i in range(len(X)):
        # x = [0 for _ in range(256)]
        # for intensity in X[i]:
        #     x[intensity] += 1
        # sample = Sample(label=Y[i], X=x)
        # samples.append(sample)
        sample = Sample(label=Y[i], X=X[i])
        samples.append(sample)

        for j in range(len(X[i])):
            if j not in attr_dict:
                attr_dict[j] = set()
            attr_dict[j].add(X[i][j])
    
    attributes = list()
    for key in attr_dict:
        attr_values = list(attr_dict[key])
        attributes.append(Attribute(name=key, values=select_attributes(attr_values, 
                                                                        min(12, len(attr_values)) # The number of attibute values to consider as possible split points. 15 ~ log2(42000)
                                                                        )
                                    )
                        )
    np.random.shuffle(samples)
    return samples, attributes