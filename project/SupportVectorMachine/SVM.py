from Utils.Model import Model
import numpy as np
from numpy.linalg import norm

class SVM(Model):
    def __init__(self, X, Y, C, tol, max_passes, k='inner_product'):
        super().__init__()
        self.X = X
        self.Y = Y
        self.alpha = np.zeroes(Y.size)
        self.b = 0
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.k = k # The kernel function, defaults to the standard inner product

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
        tot = 0
        for i in range(self.alpha.size):
            tot += self.alpha[i]*self.Y[i]*self.kernel(self.X[i], x) + self.b

    def longN(self, i, j):
        pass

    def kernel(self, x1, x2):
        """
        An SVM can only create linear decision surfaces. By expanding input data into higher dimensions,
        it may be the case that different labels associate with different dimensions, at which point a
        linear decision surface may successfully classify the data. However, transforming every data
        instance into a higher dimension may be an extremely expensive operation. To mitigate this, the
        kernel computes the inner product of the data as if it were in a transformed space. 

        Args:
            x1 (np.array): the first vector
            x2 (np.array): the second vector

        Returns:
            scalar: the inner product of the input vectors in some feature space determined by self.k
        """
        if self.k == 'inner_product':
            return np.dot(x1, x2)
        elif self.k == 'gaussian':
            sigma_sq = .25 # TODO sigma_sq could be a hyperparameter as it is constant within the routine
            nrm = norm(x1-x2, 1) # compute the 1 norm of the vectors
            return np.exp(-np.square(nrm)/sigma_sq)
        elif self.k == 'mystery':
            c = 1 # TODO 1. I don't know a good value for c, and 2. c could be a hyperparameter
            return np.square(np.dot(x1, x2) + c)
    
