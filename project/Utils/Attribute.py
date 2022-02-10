class Attribute:
    # An attribute is one of the columns in the dataset
    def __init__(self, name=None, values=None):
        self.values = values
        self.name = name


    def getValues(self):
        return self.values


    def getName(self):
        return self.name


    def setName(self, n):
        self.name = n


    def setValues(self, v):
        self.values = v


    def addValue(self, v):
        self.getValues().add(v)

        
    def __str__(self):
        return f"( name:{self.getName()} values:{self.getValues()} )"