from Utils.Model import Model
from scipy import stats
import numpy as np

class RandomForestModel(Model):
    def __init__(self):
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
        # Call learn_decision_tree()
        # Add tree to forest
        # Bag new data
        # Repeat

        # NOTE: this could be parallelized, but idc
        return -1


    def learn_decision_tree(self, examples, attributes, parent_examples):
        
        if len(examples) == 0: 
            return self.plurality_value(parent_examples)
        elif self.is_homogenous(examples):
            return examples[0].getLabel()
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            # TODO 
            # Follow the algorithm outlined on slide 9 of SupervisedLerningDecisionTrees ppt
            return -1
        

    def plurality_value(self, examples):
        labels = np.array([sample.getLabel() for sample in examples])
        return stats.mode(labels)[0]


    def is_homogenous(self, examples):
        label = examples[0].getLabel()
        for sample in examples:
            if label != sample.getLabel():
                return False
        return True
        



    