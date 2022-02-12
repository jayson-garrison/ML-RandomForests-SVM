from Utils.Model import Model
from Utils.Attribute import Attribute
from Utils.helper_functions import *
from RandomForest.Classification import *
from RandomForest.DecisionTree import *
from scipy import stats
import numpy as np

class RandomForestModel(Model):
    def __init__(self, H=entropy, maxTreeDepth=1):
        self.H = H
        self.forest = list()
        self.maxTreeDepth = maxTreeDepth
        super().__init__()


    def summary(self):
        # TODO print a summary of the trees in the learned random forest
        return -1


    def call(self, sample):
        """"
            sample: a single Sample object with a label and (vector) X
        """ 
        decisions = list()
        for tree in self.forest:
            decisions.append(tree.test_a_point(sample))
        return stats.mode(decisions, axis=None)[0][0]


    def fit(self, training_data, attributes, k):
        """
            training_data: an array of Sample objects with a label and (vector) X
            k: the the number of trees in the forest
        """
        for i in range(k):
            print(f'Constructing tree {i}')
            # sample data for building a tree (bagging)
            bag = create_bag(training_data, len(training_data))
            sample_attributes = select_attributes(attributes, int(math.sqrt(len(attributes))))
            # Call learn_decision_tree()
            stump = self.learn_decision_tree(bag, sample_attributes, list())
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
            A, threshold = argmax(self.H, attributes, examples, self.classes)
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

    
    

    


    