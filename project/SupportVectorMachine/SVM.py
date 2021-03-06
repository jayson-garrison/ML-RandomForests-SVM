from cmath import inf
from Utils.Model import Model
import numpy as np
from numpy.linalg import norm
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

class SVM(Model):
    def __init__(self, C, tol, max_passes, k='inner_product'):
        super().__init__()
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.k = k # The kernel function, defaults to the standard inner product

    def save(self):
        w = np.zeros(self.X[0].size)
        for i in range(self.alpha.size):
            w += self.alpha[i] * self.Y[i] * self.X[i]
        info = {
            'w': w,
            'b': self.b
        }
        return info

    def call(self, x):
        guess = self.f(x)
        if guess < 0: return -1
        return 1

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.alpha = np.zeros(Y.size)
        self.b = 0

        passes = 0
        log_vals = False
        prevent_overflow = True
        overflow_threshold = 10**9

        while(passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(self.alpha.size):
                E_i = self.E(i)

                # determining if KKT are violated
                if ( (self.Y[i]*E_i < -self.tol and self.alpha[i] < self.C) or \
                     (self.Y[i]*E_i > self.tol and self.alpha[i] > 0) ):

                    if log_vals: print(f'E_i={round(E_i, 4)} ', end='')
                    # select j != i randomly
                    j = i
                    while (j == i):
                        j = np.random.randint(0, self.alpha.size)
                    E_j = self.E(j)
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]
                    if log_vals: print(f'E_j={round(E_j, 4)} ', end='')
                    if log_vals: print(f'a_i_old={round(a_i_old, 4)} ', end='')
                    if log_vals: print(f'a_jold={round(a_j_old, 4)} ', end='')

                    # computing L and H using old alphas
                    if (self.Y[i] != self.Y[j]):
                        # L = max( [0, a_j_old - a_i_old] )
                        # H = min( [self.C, self.C + a_j_old - a_i_old] )
                        L = 0 if 0 > (a_j_old - a_i_old) else (a_j_old - a_i_old)
                        if L > self.C: L = self.C
                        H = self.C if self.C < self.C + a_j_old - a_i_old else self.C + a_j_old - a_i_old
                        if H < 0: H = 0
                    else:
                        # L = max( [0, a_i_old + a_j_old - self.C] )
                        # H = min( [self.C, a_i_old + a_j_old] )
                        L = 0 if 0 > (a_i_old + a_j_old - self.C) else (a_i_old + a_j_old - self.C)
                        if L > self.C: L = self.C
                        H = self.C if self.C < (a_i_old + a_j_old) else (a_i_old + a_j_old)
                        if H < 0: H = 0
                    if (L == H):
                        if log_vals: print()
                        continue

                    if L>self.C or H<0:
                        print(f'\n\na_i_old:{a_i_old}\na_j_old:{a_j_old}\nL:{L}\nH:{H}')
                        exit()

                    # get longN
                    long_n = self.longN(i, j)
                    if log_vals: print(f'long_n={round(long_n, 4)} ', end='')

                    if (long_n >= 0):
                        if log_vals: print()
                        continue
                    # compute a_j
                    a_j = a_j_old - ( (self.Y[j] * (E_i - E_j)) / long_n )
                    
                    # clip a_j
                    if (a_j > H):
                        a_j = H
                        if log_vals: print(f" a_j was CLIPPED TO H={H} ", end='')
                    elif ( a_j < L):
                        a_j = L
                        if log_vals: print(f" a_j was CLIPPED TO L={L} ", end='')
                    else:
                        if log_vals: print(" a_j was UNCLIPPED", end='')
                    # otherwise a_j is correctly computed

                    if (abs(a_j - a_j_old) < 10**-5):
                        if log_vals: print()
                        continue
                    # define new a_i
                    a_i = a_i_old + self.Y[i] * self.Y[j] * (a_j_old - a_j)
                    if log_vals: print(f'a_i={round(a_i, 4)} ', end='')
                    if log_vals: print(f'a_j={round(a_j, 4)} ', end='')
                    
                    # find b_1 and b_2
                    b_1 = self.b - E_i - self.Y[i] * (a_i - a_i_old) * self.kernel(self.X[i], self.X[i]) -\
                                         self.Y[j] * (a_j - a_j_old) * self.kernel(self.X[i], self.X[j])

                    b_2 = self.b - E_j - self.Y[i] * (a_i - a_i_old) * self.kernel(self.X[i], self.X[j]) -\
                                         self.Y[j] * (a_j - a_j_old) * self.kernel(self.X[j], self.X[j])

                    # find b
                    if (0 < a_i < self.C):
                        self.b = b_1
                    elif (0 < a_j < self.C):
                        self.b = b_2
                    else:
                        self.b = (b_1 + b_2) / 2
                    if log_vals: print(f'b_1={round(b_1, 4)} ', end='')
                    if log_vals: print(f'b_2={round(b_2, 4)} ', end='')
                    if log_vals: print(f'b={round(self.b, 4)}')
                    if log_vals or prevent_overflow: 
                        if (abs(E_i)  >= overflow_threshold or \
                            abs(E_j)  >= overflow_threshold or \
                            abs(a_i)  >= overflow_threshold or \
                            abs(a_j)  >= overflow_threshold or \
                            abs(long_n) >= overflow_threshold):
                            print(f'OVERFLOW:')
                            print(f'\n\na_i_old:{a_i_old}\na_j_old:{a_j_old}\nL:{L}\nH:{H}')
                            exit()
                    
                    self.alpha[i] = a_i
                    self.alpha[j] = a_j
                    num_changed_alphas += 1
            
            if (num_changed_alphas == 0):
                passes += 1
            else:
                passes = 0
        # pd.DataFrame(self.alpha).to_csv("alpha.csv")
        # pd.DataFrame(self.X).to_csv("X.csv")
        # pd.DataFrame(self.Y).to_csv("Y.csv")
        # print(self.b)
        # exit()


    def E(self, k):
        E_k = self.f(self.X[k]) - self.Y[k]
        return E_k

    def f(self, x):
        # TODO add self.b in every iteration?
        tot = self.b
        for i in range(self.alpha.size):
            tot += self.alpha[i]*self.Y[i]*self.kernel(self.X[i], x)
        return tot # I assume this was supposed to ret tot

    def longN(self, i, j):
        long_n = 2 * self.kernel(self.X[i], self.X[j]) - self.kernel(self.X[i], self.X[i]) -\
                                                                    self.kernel(self.X[j], self.X[j])
        return long_n

    def reset(self):
        self.alpha = np.zeros(self.Y.size)
        self.b = 0

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
    
