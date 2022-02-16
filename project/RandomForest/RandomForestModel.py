from Utils.Model import Model
from Utils.Attribute import Attribute
from Utils.helper_functions import *
from RandomForest.Classification import *
from RandomForest.DecisionTree import *
from scipy import stats
import numpy as np
import time

class RandomForestModel(Model):
    def __init__(self, H=entropy, k=20, maxTreeDepth=1, M=2):
        """
        Args:
            H ([type], optional): The information gain function. Defaults to entropy.
            k (int, optional): The number of trees in the forest. Defaults to 20.
            maxTreeDepth (int, optional): The maximum number of levels in each tree. Defaults to 1.
            M (int, optional): The number of attributes considered at each node when building the trees
        """
        self.H = H
        self.k = k
        self.maxTreeDepth = maxTreeDepth
        self.M = M
        self.forest = list()
        super().__init__()


    def summary(self):
        # TODO print a summary of the trees in the learned random forest
        return -1


    def call(self, sample):
        """Classify a point with the model
        Args:
            sample Sample: a single datapoint to be classified

        Returns:
            List of strings: each idx i is the mode classification of the first i trees in the forest
        """
        predictions = list()
        for tree in self.forest:
            predictions.append(tree.test_a_point(sample))
        decisions = list()
        for idx in range(len(predictions)):
            decisions.append(
                    stats.mode(predictions[:idx+1], axis=None)[0][0]
                )
        return decisions


    def fit(self, training_data, attributes):
        """
            training_data: an array of Sample objects with a label and (vector) X
            k: the the number of trees in the forest
        """
        self.clearForest()
        for i in range(self.k):
            # sample data for building a tree (bagging)
            bag = create_bag(training_data, len(training_data))
            # Call learn_decision_tree()
            stump = self.learn_decision_tree(bag, attributes, list())
            # Add tree to forest
            self.forest.append(stump)
            # Repeat k times

            # NOTE: this could be parallelized, but idc


    def learn_decision_tree(self, examples, attributes, parent_examples, current_depth=0):
        """
            @param: examples - a list of Sample objects
            @param: attributes - a list of Attribute objects
            @param: parent_examples - a list of Sample objects

            @return: a tree
        """
        if len(examples) == 0: 
            return self.plurality_value(parent_examples)
        elif self.is_homogenous(examples):
            return examples[0].getLabel()
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        elif current_depth == self.maxTreeDepth:
            return self.plurality_value(examples)
        else:
            sample_attributes = select_attributes(attributes, self.M)
            A, threshold = argmax(self.H, sample_attributes, examples)
            tree = DecisionTree(attribute=A)

            # Sort data based on above and below threshold
            above_samples = list()
            below_samples = list()
            for sample in examples:
                if sample.getX()[A.getName()] > threshold:
                    above_samples.append(sample)
                else:
                    below_samples.append(sample)
            
            # Make recursive call for each subset (without considering the attribute already split on)
            next_attributes = [a for a in attributes if a.getName() != A.getName()]

            # First the above values
            subtreea = self.learn_decision_tree(above_samples, next_attributes, examples, current_depth=current_depth+1)
            if not isinstance(subtreea, DecisionTree):
                subtreea = Classification(c=subtreea, label="ABOVE", threshold=threshold)
            subtreea.setLabel(f"ABOVE")
            subtreea.setThreshold(threshold)
            tree.addChild(subtreea)

            # Now the below values
            subtreeb = self.learn_decision_tree(below_samples, next_attributes, examples, current_depth=current_depth+1)
            if not isinstance(subtreeb, DecisionTree):
                subtreeb = Classification(c=subtreeb, label="BELOW", threshold=threshold)
            subtreeb.setLabel(f"BELOW")
            subtreeb.setThreshold(threshold)
            tree.addChild(subtreeb)

            return tree

        

    def plurality_value(self, examples):
        labels = np.array([sample.getLabel() for sample in examples])
        return stats.mode(labels, axis=None)[0][0]


    def is_homogenous(self, examples):
        label = examples[0].getLabel()
        for sample in examples:
            if label != sample.getLabel():
                return False
        return True

    def getK(self):
        return self.k
    
    def clearForest(self):
        self.forest = list()

    
    

    


    