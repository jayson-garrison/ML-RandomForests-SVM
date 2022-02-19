import numpy as np
import time

def pca(array, n): # take in an array, return a transformed array with n dimensions
    start = time.time()
    print("Entering PCA ... ", end='')
    print(f"Standardizing {int(time.time()-start)} ... ", end='')
    array = standardize(array)
    print(f"Finding Covariance {int(time.time()-start)} ... ", end='')
    cov = np.cov(array.T)
    print(f"Finding Eigens {int(time.time()-start)} ... ", end='')
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    print(f"Projecting {int(time.time()-start)} ... ", end='')
    projector = (eigen_vectors.T[:][:n]).T
    projected = array.dot(projector)
    print(f'Exiting PCA, Total Elapsed Time: {time.time()-start}')
    return projected

def standardize(array): # take in an array and for every standardize every feature dimension
    rows, columns = array.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(array[:,column])
        std = max(np.std(array[:,column]), .001)
        tempArray = np.empty(0)
        
        for element in array[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray

