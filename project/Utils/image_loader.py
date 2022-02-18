import pandas as pd
import numpy as np
from Utils.Sample import Sample
from Utils.Attribute import Attribute
from Utils.helper_functions import *

def load_image_data():
    # Restructure the data so that |X| is 255 (one feature for each intensity) and x_i is the count for the ith intensity in the image X_i
    data = pd.read_csv('./project/Datasets/mnist/train.csv')
    data = data.to_numpy()
    np.random.shuffle(data)
    X = data[1:, 1:]
    Y = data[1:, 0]
    samples = list()
    attr_dict = dict()
    for i in range(len(X)):
        x = [0 for _ in range(256)]
        for intensity in X[i]:
            x[intensity] += 1
        sample = Sample(label=Y[i], X=x)
        samples.append(sample)

        for j in range(len(x)):
            if j not in attr_dict:
                attr_dict[j] = set()
            attr_dict[j].add(x[j])
    
    attributes = list()
    for key in attr_dict:
        attr_values = list(attr_dict[key])
        attributes.append(Attribute(name=key, values=select_attributes(attr_values, 
                                                                        15 # The number of attibute values to consider as possible split points. 15 ~ log2(42000)
                                                                        )
                                    )
                        )
    np.random.shuffle(samples)
    return samples, attributes