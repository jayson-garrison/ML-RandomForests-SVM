""""
Author: Jayson C. Garrison
Dates: 02/11/2022
Course: CS-5333 (Machine Learning)
GitHub: https://github.com/jayson-garrison/ML-Naive-Bayes
"""
import numpy as np
import time 

# partition a data set into a list of 5 tuples for training and testing
# five fold data partition
def five_fold(data_set):
    """[summary]

    Args:
        data_set (List of Sample objects): The Samples to be partitioned

    Returns:
        fold: where fold is list of len n in n-fold of (train,test) where train and test are lists of Samples
    """
    partition_index = int( len(data_set) / 5 )
    s = 0
    fold = []
    for i in range(5): #0-4
        tr = data_set.copy()
        n = s + partition_index # was -1
        te = tr[s:n]
        del tr[s:s + partition_index]

        fold.append( (tr,te) )

        s += partition_index

    return fold


def make_output_strings_rf(dataset, model, attributes):
    fold = five_fold(dataset)
    output = [f'{i+1},' for i in range(model.getK())]
    train_columns = ['' for _ in range(len(output))]
    test_columns = ['' for _ in range(len(output))]
    validation_count = 0
    for partition in fold:
        start = time.time()
        validation_count += 1
        print(f'Fitting model [{validation_count}/5]', end=' ... ')
        model.fit(partition[0], attributes)
        print(f'Elapsed Time: {time.time()-start}')
        start = time.time()
        print(f'Validating model [{validation_count}/5]', end=' ... ')
        # First compute the training accuracies
        train_accuracies = [0 for _ in range(len(output))]
        for sample in partition[0]:
            predictions = model.call(sample) # A list of k predictions if there are k trees, the ith prediction is the mode guess of the first i trees
            label = sample.getLabel()
            for idx in range(len(predictions)):
                if label == predictions[idx]:
                    train_accuracies[idx] += 1
        for idx in range(len(train_accuracies)):
            train_accuracies[idx] = train_accuracies[idx]/len(partition[0])

        
        # Now compute the testing accuracies
        test_accuracies = [0 for _ in range(len(output))]
        for sample in partition[1]:
            predictions = model.call(sample)
            label = sample.getLabel()
            for idx in range(len(predictions)):
                if label == predictions[idx]:
                    test_accuracies[idx] += 1
        for idx in range(len(test_accuracies)):
            test_accuracies[idx] = test_accuracies[idx]/len(partition[1])

        # print(f'train_acc: {train_accuracies[-1]}, test_acc: {test_accuracies[-1]}')
        # model.forest[0].pretty_print_tree()
        # exit()
        
        # Finally, build the output
        for idx in range(len(output)):
            train_columns[idx] += str(train_accuracies[idx])+','
            test_columns[idx] += str(test_accuracies[idx])+','
        print(f'Elapsed Time: {time.time()-start}')

    # Aggregate the column information
    for idx in range(len(output)):
        output[idx] += train_columns[idx] + test_columns[idx] + '\n'

    return output


# NOTE: please ignore the fact that this is a virtual copy of make_output_strings_rf()
def make_output_strings_svm(dataset, model):
    fold = five_fold(dataset)
    output = [f'{i+1},' for i in range(model.getK())]
    train_columns = ['' for _ in range(len(output))]
    test_columns = ['' for _ in range(len(output))]
    validation_count = 0
    for partition in fold:
        start = time.time()
        validation_count += 1
        print(f'Fitting model [{validation_count}/5]', end=' ... ')
        model.fit()
        print(f'Elapsed Time: {time.time()-start}')
        start = time.time()
        print(f'Validating model [{validation_count}/5]', end=' ... ')
        # First compute the training accuracies
        train_accuracies = [0 for _ in range(len(output))]
        for sample in partition[0]:
            predictions = model.call(sample) # A list of k predictions if there are k trees, the ith prediction is the mode guess of the first i trees
            label = sample.getLabel()
            for idx in range(len(predictions)):
                if label == predictions[idx]:
                    train_accuracies[idx] += 1
        for idx in range(len(train_accuracies)):
            train_accuracies[idx] = train_accuracies[idx]/len(partition[0])

        
        # Now compute the testing accuracies
        test_accuracies = [0 for _ in range(len(output))]
        for sample in partition[1]:
            predictions = model.call(sample)
            label = sample.getLabel()
            for idx in range(len(predictions)):
                if label == predictions[idx]:
                    test_accuracies[idx] += 1
        for idx in range(len(test_accuracies)):
            test_accuracies[idx] = test_accuracies[idx]/len(partition[1])

        # print(f'train_acc: {train_accuracies[-1]}, test_acc: {test_accuracies[-1]}')
        # model.forest[0].pretty_print_tree()
        # exit()
        
        # Finally, build the output
        for idx in range(len(output)):
            train_columns[idx] += str(train_accuracies[idx])+','
            test_columns[idx] += str(test_accuracies[idx])+','
        print(f'Elapsed Time: {time.time()-start}')

    # Aggregate the column information
    for idx in range(len(output)):
        output[idx] += train_columns[idx] + test_columns[idx] + '\n'

    return output

            


        




