import pandas as pd
import numpy as np
from Utils.Sample import Sample
from Utils.Attribute import Attribute
from Utils.PCA import *
from Utils.helper_functions import select_attributes


def load_email_data():
    data = pd.read_csv('./project/Datasets/spam_ham/pca_emails.csv')
    data = data.to_numpy()
    X = data[1:, 1:-1]
    #X = pca(X, 4)
    Y = data[1:, -1]

    #pca_emails = pd.DataFrame(np.column_stack((Y,X))) 
    #pca_emails.to_csv('pca_emails.csv')
    
    samples = list()
    attr_dict = dict()
    for i in range(int(len(X))):
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
                                                                        12 # The number of attibute values to consider as possible split points.
                                                                        )
                                    )
                        )
    # print(f'DIMS POST PCA: {len(samples)} x {len(samples[0].getX())}')
    np.random.shuffle(samples)
    return samples, attributes
