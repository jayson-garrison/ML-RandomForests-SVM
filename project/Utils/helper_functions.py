import math
import random as rand
import numpy as np

def argmax(evaluation_function, attributes, samples):
    """
        evaluation_function : either entropy, gini, or misclassifictaion
        args1 : a list of Attribute objects
        args2 : the list of Sample objects
        return : the Attribute and the Threshold value for splitting
    """
    m = float('-inf')
    chosen_attr = attributes[0]
    chosen_threshold = 0
    all_entropies = list()
    for attribute in attributes: 
        for thr in attribute.getValues():
            r = evaluation_function(attribute, samples, thr)
            all_entropies.append(r)

            if r > m: 
                m = r
                chosen_attr = attribute
                chosen_threshold = thr   
    return chosen_attr, chosen_threshold

def misclassification(attribute, samples, threshold):
    # TODO check these computations
    # sort the samples into bins by their value for the attribute
    above_samples = set()
    above_class_cts = dict()
    below_samples = set()
    below_class_cts = dict()
    for sample in samples:
        if sample.getX()[attribute.getName()] > threshold:
            above_samples.add(sample)
            c = sample.getLabel()
            if c not in above_class_cts:
                above_class_cts[c] = 0
            above_class_cts[c] += 1
        else:
            below_samples.add(sample)
            c = sample.getLabel()
            if c not in below_class_cts:
                below_class_cts[c] = 0
            below_class_cts[c] += 1
    values = list()
    count = 0

    for possible in above_class_cts:
        for item in above_samples:
            if possible == item:
                count += 1
        values.append(count / len(above_samples))
        count = 0

    count = 0
    for possible in below_class_cts:
        for item in below_samples:
            if possible == item:
                count += 1
        values.append(count / len(below_samples))
        count = 0

    return 1 - max(values)

def entropy(attribute, samples, threshold):
    # TODO check these computations
    # sort the samples into bins by their value for the attribute
    above_samples = set()
    above_class_cts = dict()
    below_samples = set()
    below_class_cts = dict()
    for sample in samples:
        if sample.getX()[attribute.getName()] > threshold:
            above_samples.add(sample)
            c = sample.getLabel()
            if c not in above_class_cts:
                above_class_cts[c] = 0
            above_class_cts[c] += 1
        else:
            below_samples.add(sample)
            c = sample.getLabel()
            if c not in below_class_cts:
                below_class_cts[c] = 0
            below_class_cts[c] += 1

    # now bins has the counts for each value, run entropy calculating p
    N = len(samples)
    atot = 0
    na = len(above_samples)
    for c in above_class_cts:
        pcabove = above_class_cts[c] / na
        atot += pcabove * math.log2(pcabove)
    btot = 0
    nb = len(below_samples)
    for c in below_class_cts:
        pcbelow = below_class_cts[c] / nb
        btot += pcbelow * math.log2(pcbelow)
    
    weighted_total = atot * (na/N) + btot * (nb/N)
    
    return weighted_total


def gini(attribute, samples, threshold):
    # sort the samples into bins by their value for the attribute
    above_samples = set()
    above_class_cts = dict()
    below_samples = set()
    below_class_cts = dict()
    for sample in samples:
        if sample.getX()[attribute.getName()] > threshold:
            above_samples.add(sample)
            c = sample.getLabel()
            if c not in above_class_cts:
                above_class_cts[c] = 0
            above_class_cts[c] += 1
        else:
            below_samples.add(sample)
            c = sample.getLabel()
            if c not in below_class_cts:
                below_class_cts[c] = 0
            below_class_cts[c] += 1

    # now bins has the counts for each value, run entropy calculating p
    N = len(samples)
    atot = 0
    na = len(above_samples)
    for c in above_class_cts:
        pcabove = above_class_cts[c] / na
        atot += pcabove * (1 - pcabove)
    btot = 0
    nb = len(below_samples)
    for c in below_class_cts:
        pcbelow = below_class_cts[c] / nb
        btot += pcbelow * (1 - pcbelow)
    
    weighted_total = atot * (na/N) + btot * (nb/N)

    return -1* weighted_total

def create_bag(samples, n):
    '''
    @param samples, the aggregate data set
    @param n, the cardinality of the bag
    @return the bag
    '''
    bag  = rand.choices(samples, k=n)
    return bag

def select_attributes(attributes, n):
    selected_attributes = np.random.choice(attributes, n, replace=False)
    return selected_attributes

