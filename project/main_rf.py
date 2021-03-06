from fileinput import filename
import os
from tkinter import E
from RandomForest.RandomForestModel import *
from Utils.email_loader import load_email_data
from Utils.image_loader import load_image_data
from Utils.artificial_data_loader import *
from Utils.five_fold import *
from Utils.visualize_log import *
import time


if __name__ == "__main__":    
    # y = np.array((0, 1, 2))
    # x = np.matrix([[0, 0], [1, 1], [2, 2]])
    # print(np.column_stack((y, x)))
    # exit()

    run_analysis = False
    generate_visuals = True

    # Blobs only has two attributes, M=2 always
    params_blobs = [
        #(1, entropy, 'entropy', 'blob', 2),
        #(1, gini, 'gini', 'blob', 2),
        (1, misclassification, 'misclassification', 'blob', 2),
        #(2, entropy, 'entropy', 'blob', 2),
        #(2, gini, 'gini', 'blob', 2),
        (2, misclassification, 'misclassification', 'blob', 2),
        #(3, entropy, 'entropy', 'blob', 2),
        #(3, gini, 'gini', 'blob', 2),
        (3, misclassification, 'misclassification', 'blob', 2),
    ]
    # Spiral only has two attributes, M=2 always
    params_spiral = [
        (1, entropy, 'entropy', 'spiral', 2),
        (1, gini, 'gini', 'spiral', 2),
        (1, misclassification, 'misclassification', 'spiral', 2),
        (2, entropy, 'entropy', 'spiral', 2),
        (2, gini, 'gini', 'spiral', 2),
        (2, misclassification, 'misclassification', 'spiral', 2),
        (3, entropy, 'entropy', 'spiral', 2),
        (3, gini, 'gini', 'spiral', 2),
        (3, misclassification, 'misclassification', 'spiral', 2)]
    # Image has 3000 attributes, 10 after pca. Let M be in {2, 3, 4}
    params_image = [
        # (4, entropy, 'entropy', 'image', 2),
        # (4, entropy, 'entropy', 'image', 3),
        # (4, entropy, 'entropy', 'image', 4),
        # (4, gini, 'gini', 'image', 2),
        # (4, gini, 'gini', 'image', 3),
        # (4, gini, 'gini', 'image', 4),
        # (4, misclassification, 'misclassification', 'image', 2),
        # (4, misclassification, 'misclassification', 'image', 3),
        # (4, misclassification, 'misclassification', 'image', 4),

        #(5, entropy, 'entropy', 'image', 2),
        (5, entropy, 'entropy', 'image', 3),
        #(5, entropy, 'entropy', 'image', 4),
        #(5, gini, 'gini', 'image', 2),
        (5, gini, 'gini', 'image', 3),
        #(5, gini, 'gini', 'image', 4),
        #(5, misclassification, 'misclassification', 'image', 2),
        (5, misclassification, 'misclassification', 'image', 3),
        #(5, misclassification, 'misclassification', 'image', 4),
        #(6, entropy, 'entropy', 'image', 2),
        (6, entropy, 'entropy', 'image', 3),
        #(6, entropy, 'entropy', 'image', 4),
        #(6, gini, 'gini', 'image', 2),
        (6, gini, 'gini', 'image', 3),
        #(6, gini, 'gini', 'image', 4),
        #(6, misclassification, 'misclassification', 'image', 2),
        (6, misclassification, 'misclassification', 'image', 3),
        #(6, misclassification, 'misclassification', 'image', 4),
    ]
    # Mail has 256 attributes, 4 after pca. Let M be in {2, 3, 4}
    params_mail = [
        #(1, entropy, 'entropy', 'mail', 2),
        #(1, entropy, 'entropy', 'mail', 4),
        #(1, entropy, 'entropy', 'mail', 3),
        #(1, gini, 'gini', 'mail', 2),
        #(1, gini, 'gini', 'mail', 4),
        #(1, gini, 'gini', 'mail', 3),
        #(1, misclassification, 'misclassification', 'mail', 2),
        #(1, misclassification, 'misclassification', 'mail', 4),
        #(1, misclassification, 'misclassification', 'mail', 3),
        (2, entropy, 'entropy', 'mail', 2),
        (2, entropy, 'entropy', 'mail', 4),
        #(2, entropy, 'entropy', 'mail', 3),
        (2, gini, 'gini', 'mail', 2),
        (2, gini, 'gini', 'mail', 4),
        #(2, gini, 'gini', 'mail', 3),
        (2, misclassification, 'misclassification', 'mail', 2),
        (2, misclassification, 'misclassification', 'mail', 4),
        #(2, misclassification, 'misclassification', 'mail', 3),
        (3, entropy, 'entropy', 'mail', 2),
        (3, entropy, 'entropy', 'mail', 4),
        #(3, entropy, 'entropy', 'mail', 3),
        (3, gini, 'gini', 'mail', 2),
        (3, gini, 'gini', 'mail', 4),
        #(3, gini, 'gini', 'mail', 3),
        (3, misclassification, 'misclassification', 'mail', 2),
        (3, misclassification, 'misclassification', 'mail', 4),
        #3, misclassification, 'misclassification', 'mail', 3),
        
    ]
    
    for params in params_image:
        # params = (3, gini, 'gini', 'mail', 2)
        hyper_parameters = {
            'mdt': params[0], # Max depth of the tree
            'h': params[1], # The evaluation function
            'hname': params[2], # String name of the evaluation function
            'dataset': params[3], # data sets, 0: blob, 1: spiral, 2: mail, 3: image
            'M': params[4], # the number of Attributes considered at each node during tree construction
            'nt': 150 # the total number of trees in the forest
        }
        filename = 'project/Logs/RF/DataSet_'+str(hyper_parameters['dataset'])+'/'+\
                                '_NumTrees-'+str(hyper_parameters['nt'])+\
                                '_MaxDepth-'+str(hyper_parameters['mdt'])+\
                                '_InfoGain-'+str(hyper_parameters['hname'])+\
                                '_M-'+str(hyper_parameters['M'])+'.txt'
        
        print(f'Creating file {filename}')
        if run_analysis:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as log:
                log.write('num,train1,train2,train3,train4,train5,test1,test2,test3,test4,test5\n')
                rf = RandomForestModel(H=hyper_parameters['h'], 
                                    k=hyper_parameters['nt'],
                                    maxTreeDepth=hyper_parameters['mdt'], 
                                    M=hyper_parameters['M'])
                if hyper_parameters['dataset'] == 'blob':
                    dataset, attributes = load_artificial_data('./project/Datasets/artificial_datasets/dataset/blobs.csv')
                elif hyper_parameters['dataset'] == 'spiral':
                    dataset, attributes = load_artificial_data('./project/Datasets/artificial_datasets/dataset/spirals.csv')
                elif hyper_parameters['dataset'] == 'mail':
                    dataset, attributes = load_email_data()
                elif hyper_parameters['dataset'] == 'image':
                    dataset, attributes = load_image_data()
                else:
                    print('No data set given..')
                    exit()
                output = make_output_strings_rf(dataset, rf, attributes)
                for line in output:
                    log.write(line)
                log.close()
                # exit()

    if generate_visuals:
        log_list1 = [
            '_NumTrees-150_MaxDepth-3_InfoGain-entropy_M-2.txt',
            '_NumTrees-150_MaxDepth-3_InfoGain-gini_M-2.txt',
            '_NumTrees-150_MaxDepth-3_InfoGain-misclassification_M-2.txt'
        ]
        log_list_image1 = [
            'DataSet_image/_NumTrees-150_MaxDepth-5_InfoGain-entropy_M-3.txt',
            'DataSet_image/_NumTrees-150_MaxDepth-5_InfoGain-gini_M-3.txt',
            'DataSet_image/_NumTrees-150_MaxDepth-5_InfoGain-misclassification_M-3.txt'
        ]
        log_list_image2 = [
            'DataSet_image/_NumTrees-150_MaxDepth-6_InfoGain-entropy_M-3.txt',
            'DataSet_image/_NumTrees-150_MaxDepth-6_InfoGain-gini_M-3.txt',
            'DataSet_image/_NumTrees-150_MaxDepth-6_InfoGain-misclassification_M-3.txt'
        ]
        log_list_mail1 = [
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-entropy_M-2.txt',
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-gini_M-2.txt',
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-misclassification_M-2.txt'
        ]
        log_list_mail2 = [
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-entropy_M-4.txt',
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-gini_M-4.txt',
            'DataSet_mail/_NumTrees-150_MaxDepth-3_InfoGain-misclassification_M-4.txt'
        ]
        visualize_log(log_list_mail2, 'DataSet_mail/', 'Performance on Depth 3 and M=4 on Mail')
        exit()
        log_dir = 'project/Logs/RF/'
        for file in os.listdir(log_dir):
            visualize_log(list(file), 'DataSet_image/',file)
        # visualize_log(filename)
        