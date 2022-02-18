# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_log(log_files, data_name, graph_name):
    """
    plots the log file's train and testing performance
    @param log_files, list of files to be displayed in the same graph
    @param data_name, the name of the data set with /
    @param graph_name, the name of the graph
    """
    measures = ['Entropy', 'Gini', 'Misclassification']
    counter = 0
    for log in log_files:
        data = pd.read_csv('project/Logs/RF/'+data_name +log)
        data = data.to_numpy()
        X1 = data[:, 1:6]
        Y = np.array(list(range(150))) # will need to be a variable for generality
        X2 = data[:, 6:-1]

        x_train = list()
        for array in X1:
            x_train.append(np.mean(array))
        
        x_test = list()
        for array in X2:
            x_test.append(np.mean(array))

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)

        #print(x_test)

        plt.plot(Y,x_train, label = measures[counter]+' Train')
        plt.plot(Y,x_test, label = measures[counter]+' Test')
        plt.title(graph_name)
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor = (.5, .6))
        plt.savefig('project/Visuals/RF/'+data_name+graph_name+'.png')
        counter += 1
        #plt.show()

    plt.close()

    
    
     