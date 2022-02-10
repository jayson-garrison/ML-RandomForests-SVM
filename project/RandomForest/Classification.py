class Classification:
    # The leaf nodes of the decision tree
    def __init__(self, c=0, label="NONE", threshold=None):
        self.c = c
        self.label = label
        self.threshold = threshold

    def getClass(self):
        return self.c

    def getLabel(self):
        return self.label

    def setClass(self, c):
        self.c = c

    def setLabel(self, l):
        self.label = l

    def getThreshold(self):
        return self.threshold
        
    def setThreshold(self, t):
        self.threshold = t