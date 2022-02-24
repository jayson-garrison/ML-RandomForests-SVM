# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
        data = pd.read_csv('project/Logs/RF/'+log)
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
        #plt.legend(bbox_to_anchor = (.5, .6))
        plt.legend()
        plt.savefig('project/Visuals/RF/'+data_name+graph_name+'.png')
        counter += 1
        #plt.show()

    plt.close()

    

def visualize_log_svm(dir, files, graph_name):
    counter = 0
    avg_train_validations = list()
    avg_test_validations = list()
    Cs = np.array([.01, .05, .1, .5, 1, 10, 100, 1000])

    for file in files:
        data = pd.read_csv('project/Logs/SVM/'+dir+'/'+file)
        data = data.to_numpy()
        X1 = data[:, 1:6] # train
        X2 = data[:, 6:-1] # test

        for array in X1:
            avg_train_validations.append(np.mean(array))
        
        for array in X2:
            avg_test_validations.append(np.mean(array))
    
    avg_train_validations = np.array(avg_train_validations)
    avg_test_validations = np.array(avg_test_validations)

    # print(f'Cs: {Cs} and tr: {avg_train_validations}')
    # exit()

    plt.plot(Cs.astype('str'),avg_train_validations, label = 'Train')
    plt.plot(Cs.astype('str'),avg_test_validations, label = 'Test')
    plt.title(graph_name)
    plt.xlabel('C-Value')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor = (.5, .6))
    plt.savefig('project/Visuals/SVM/'+dir+'/'+graph_name+'.png')
    counter += 1
    plt.show()

    plt.close()

if __name__ == "__main__":
    # print(os.listdir('project/Logs/SVM/DataSet_spiral'))
    valid_files = [
                    '_Softness-0.01_Tolerance-0.1_Kernel-gaussian.txt', 
                    '_Softness-0.05_Tolerance-0.1_Kernel-gaussian.txt', 
                    '_Softness-0.1_Tolerance-0.1_Kernel-gaussian.txt', 
                    '_Softness-0.5_Tolerance-0.1_Kernel-gaussian.txt', 
                    '_Softness-1_Tolerance-0.1_Kernel-gaussian.txt', 
                    '_Softness-10_Tolerance-0.1_Kernel-gaussian.txt',
                    '_Softness-100_Tolerance-0.1_Kernel-gaussian.txt',
                    '_Softness-1000_Tolerance-0.1_Kernel-gaussian.txt',
                    ]
    graph_name = 'SVM on Blobs with Gaussian Kernel and Tolerance 0.1'
    visualize_log_svm('DataSet_blobs', valid_files, graph_name)

    