from re import L
import pandas as pd
import numpy as np
from sklearn.utils import indices_to_mask
from Utils.Sample import Sample
from Utils.Attribute import Attribute
from Utils.PCA import *
from Utils.helper_functions import select_attributes


def load_email_data(using_svm=False):
    data = pd.read_csv('./project/Datasets/spam_ham/pca_emails.csv')
    data = data.to_numpy()
    X = data[1:, 2:]
    #X = pca(X, 4)
    Y = data[1:, 1]
    #pca_emails = pd.DataFrame(np.column_stack((Y,X))) 
    #pca_emails.to_csv('pca_emails.csv')

    #========================================================
    # Uncomment this routine to keep only a subset of the samples
    num_samples = 500
    indices = set()
    while(len(indices) < num_samples):
        indices.add(np.random.randint(len(Y)))
    indices = list(indices)
    indices.sort()
    X = np.take(X, indices, axis=0)
    X = norm_0_1(X)
    Y = np.take(Y, indices, axis=0)
    #========================================================

    
    samples = list()
    attr_dict = dict()
    for i in range(int(len(X))):
        if Y[i] == 0: Y[i] = -1 # NOTE: This change is necessary for SVM
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
                                                                        min(12, len(attr_values)) # The number of attibute values to consider as possible split points.
                                                                        )
                                    )
                        )
    # print(f'DIMS POST PCA: {len(samples)} x {len(samples[0].getX())}')
    np.random.shuffle(samples)

    if using_svm: return samples, attributes, X, Y
    return samples, attributes
