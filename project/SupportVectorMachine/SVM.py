from Utils.Model import Model
import numpy as np

class SVM(Model):
    def __init__(self, X, Y, C, tol, max_passes):
        super().__init__()
        self.X = X
        self.Y = Y
        self.alpha = np.zeroes(Y.size)
        self.b = 0
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def call(self):
        pass

    def fit(self):
        passes = 0
        while(passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(self.alpha.size):
                Ei = self.E(i)
                if (self.Y[i]*Ei < -self.tol and self.alpha[i] < self.C) or (self.Y[i]*Ei>self.tol and self.alpha[i]>0):
                    # select j != i randomly
                    j = i
                    while (j == i):
                        j = np.random.randint(0, self.alpha.size)
                    Ej = self.E(j)
                    aiold = self.alpha[i]
                    ajold = self.alpha[j]
                    


    def E(self, i):
        pass

    def f(self, x):
        pass

    def longN(self, i, j):
        pass

    
