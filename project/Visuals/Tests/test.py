import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from Utils.artificial_data_loader import *



b = -1.6609416873391798
alpha = pd.read_csv('alpha.csv').to_numpy()[:, 1]
X = pd.read_csv('X.csv').to_numpy()[:, 1:]
Y = pd.read_csv('Y.csv').to_numpy()[:, 1]
# alpha_str = "0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.77555756e-17 0.00000000e+00  0.00000000e+00  0.00000000e+00 -1.11022302e-16 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 5.55111512e-17  4.33680869e-18  0.00000000e+00  0.00000000e+00 0.00000000e+00 -5.55111512e-17  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.74484834e+00 -1.38777878e-17 -1.52655666e-16  0.00000000e+00 -1.04083409e-17 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.91091694e+00 4.30351492e+00  1.34171172e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  3.33066907e-16  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.77555756e-17  0.00000000e+00  0.00000000e+00  8.31602341e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 1.38777878e-17  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00 -2.22044605e-16  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  1.79268403e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00 -1.11022302e-16  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.08166817e-17  0.00000000e+00 -3.46944695e-18  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 6.07153217e-17  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00 -5.55111512e-17 0.00000000e+00  0.00000000e+00  6.93889390e-18  1.00000000e+01 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 2.57805626e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  2.56057874e+00  0.00000000e+00 -1.38777878e-17 -2.77555756e-17  1.65500089e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  5.55111512e-17 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 -4.16333634e-17  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  4.44683662e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.61001847e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 1.11022302e-16  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  2.77555756e-17  0.00000000e+00  0.00000000e+00 1.90819582e-17  2.77555756e-17  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 -1.38777878e-17  0.00000000e+00  0.00000000e+00 -6.54858112e-17 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 2.77555756e-17  8.32667268e-17  0.00000000e+00  3.46944695e-17 0.00000000e+00  2.71523921e+00  1.99867496e+00  0.00000000e+00 0.00000000e+00  4.44449225e-01  3.69334683e-03"
# alpha_str_arr = np.array(alpha_str.split())
# alpha = alpha_str_arr.astype('float')
# dataset, attributes, X, Y = load_artificial_data('./project/Datasets/artificial_datasets/dataset/spirals.csv', using_svm=True)
# def convert_range(x, input_start, input_end, output_start, output_end):
#     out = (x - input_start) / (input_end - input_start) * (output_end - output_start) + output_start
#     return out

print(alpha.shape)
print(X.shape)
print(Y.shape)


def gauss(x_1, x_2):
    sigma_sq = .25 # TODO sigma_sq could be a hyperparameter as it is constant within the routine
    nrm = norm(x_1-x_2, 1) # compute the 1 norm of the vectors
    return np.exp(-np.square(nrm)/sigma_sq)

def f(x):
    tot = b
    for i in range(alpha.size):
        tot += alpha[i]*Y[i]*gauss(X[i], x)
    return tot

def validate(x, y):
    correct = 0
    total = 0
    for idx in range(len(y)):
        prediction = -1 if f(x[idx]) < 0 else 1
        label = y[idx]
        if label == prediction:
            correct += 1
        total += 1
    return correct/total

print(f'validation: {validate(X, Y)}')

start = -1.5
stop = 1.5
step = .05
x1 = np.arange(start, stop, step)
x2 = np.arange(start, stop, step)
y_fin = np.zeros(shape=(int(abs(stop-start)/step), int(abs(stop-start)/step)))
for row in range(x1.size):
    for col in range(x2.size):
        y_fin[row][col] = -1 if f(np.array([x1[row],x2[col]])) < 0 else 1



# heatmap2d(y_fin)

fig, ax = plt.subplots()
ax.imshow(y_fin)

plt.show()