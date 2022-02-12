# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_log(log_file):
    """
    plots the log file's train and testing performance
    @param log_file, the name of the file to be visualized
    """
    data = pd.read_csv(log_file)
    data = data.to_numpy()
    X1 = data[:, 1:6]
    Y = data[:, 0]
    X2 = data[:, 6:-1]

    x_train = list()
    for array in X1:
        x_train.append(np.mean(array))
    
    x_test = list()
    for array in X2:
        x_test.append(np.mean(array))

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    print(x_test)

    plt.plot(Y,x_train, label = 'Train')
    plt.plot(Y,x_test, label = 'Test')
    plt.title('DEFAULT')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('NAME.png')
    
     