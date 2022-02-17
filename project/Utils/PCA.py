import numpy as np
import time

def pca(array, n): # take in an array, return a transformed array with n dimensions
    start = time.time()
    print("Entering PCA ... ", end='')
    array = standardize(array)
    cov = np.cov(array.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    projection_matrix = (eigen_vectors.T[:][:n]).T
    X_pca = array.dot(projection_matrix)
    print(f'Exiting PCA, Elapsed Time: {time.time()-start}')
    return X_pca

def standardize(array): # take in an array and for every standardize every feature dimension
    rows, columns = array.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(array[:,column])
        std = np.std(array[:,column])
        tempArray = np.empty(0)
        
        for element in array[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray

