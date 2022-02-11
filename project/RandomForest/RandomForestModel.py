from Utils.Model import Model
from Classification import *
from scipy import stats
import numpy as np

class RandomForestModel(Model):
    def __init__(self, H='entropy'):
        self.H = H
        self.forest = list()
        super().__init__()


    def summary(self):
        # TODO print a summary of the trees in the learned random forest
        return -1


    def call(self, sample):
        """"
            sample: a single Sample object with a label and (vector) X
        """
        # TODO 
        return -1


    def fit(self, training_data, k):
        """
            training_data: an array of Sample objects with a label and (vector) X
            k: the the number of trees in the forest
        """
        # TODO
        # sample data for building a tree (bagging)
        # Call learn_decision_tree()
        # Add tree to forest
        # Repeat k times

        # NOTE: this could be parallelized, but idc
        return -1


    def learn_decision_tree(self, examples, attributes, parent_examples):
        """
            @param: examples - a list of Sample objects
            @param: attributes - a set of integer values
            @param: parent_examples - a list of Sample objects

            @return: a tree
        """
        if len(examples) == 0: 
            return self.plurality_value(parent_examples)
        elif self.is_homogenous(examples):
            return examples[0].getLabel()
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            # TODO 
            # Follow the algorithm outlined on slide 9 of SupervisedLearningDecisionTrees ppt
            return -1
        

    def plurality_value(self, examples):
        labels = np.array([sample.getLabel() for sample in examples])
        return stats.mode(labels, axis=None)[0][0]


    def is_homogenous(self, examples):
        label = examples[0].getLabel()
        for sample in examples:
            if label != sample.getLabel():
                return False
        return True

    
    def entropy(attribute, samples, threshold):
        # TODO check these computations
        # sort the samples into bins by their value for the attribute
        above_samples = set()
        above_class_cts = dict()
        below_samples = set()
        below_class_cts = dict()
        for s in samples:
            if s.getValue(attribute.getName()) > threshold:
                above_samples.add(s)
                c = s.getClassification()
                if c not in above_class_cts:
                    above_class_cts[c] = 0
                above_class_cts[c] += 1
            else:
                below_samples.add(s)
                c = s.getClassification()
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
        for s in samples:
            if s.getValue(attribute.getName()) > threshold:
                above_samples.add(s)
                c = s.getClassification()
                if c not in above_class_cts:
                    above_class_cts[c] = 0
                above_class_cts[c] += 1
            else:
                below_samples.add(s)
                c = s.getClassification()
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

    
    def test_a_point(decision_tree, sample):
        # Put a point into the tree, whether the classification was correct or not
        node = decision_tree
        while not isinstance(node, Classification):
            attr = node.getAttribute().getName()
            for child in node.getChildren():
                if (sample.getValue(attr) > child.getThreshold()) and (child.getLabel()=="ABOVE"):
                    node = child
                    break
                elif (sample.getValue(attr) <= child.getThreshold()) and (child.getLabel()=="BELOW"):
                    node = child
                    break
            
        return node.getClass()==sample.getClassification()

    


    