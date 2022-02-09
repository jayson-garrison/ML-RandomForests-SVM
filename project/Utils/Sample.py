import numpy as np

class Sample():
    def __init__(self, label, X):
        self.label = label
        self.X = np.array(X)

    def getLabel(self):
        return self.label

    def getX(self):
        return self.X