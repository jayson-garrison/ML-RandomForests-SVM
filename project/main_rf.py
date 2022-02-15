from fileinput import filename
import os
from RandomForest.RandomForestModel import *
from Utils.email_loader import load_email_data
from Utils.image_loader import load_image_data
from Utils.artificial_data_loader import *
from Utils.five_fold import *
from Utils.visualize_log import *


if __name__ == "__main__":
    run_analysis = True
    params_list_0 = [
        (1, 100, entropy, 'entropy', 'spiral'),
        (1, 100, gini, 'gini', 'spiral'),
        (1, 100, misclassification, 'misclassification', 'spiral'),
        (2, 100, entropy, 'entropy', 'spiral'),
        (2, 100, gini, 'gini', 'spiral'),
        (2, 100, misclassification, 'misclassification', 'spiral'),
        (3, 100, entropy, 'entropy', 'spiral'),
        (3, 100, gini, 'gini', 'spiral'),
        (3, 100, misclassification, 'misclassification', 'spiral'),
        (1, 100, entropy, 'entropy', 'image'),
        (1, 100, gini, 'gini', 'image'),
        (1, 100, misclassification, 'misclassification', 'image'),
        (2, 100, entropy, 'entropy', 'image'),
        (2, 100, gini, 'gini', 'image'),
        (2, 100, misclassification, 'misclassification', 'image'),
        (3, 100, entropy, 'entropy', 'image'),
        (3, 100, gini, 'gini', 'image'),
        (3, 100, misclassification, 'misclassification', 'image'),
    ]
    params_list_1 = [
        (1, 100, entropy, 'entropy', 'mail'),
        (1, 100, gini, 'gini', 'mail'),
        (1, 100, misclassification, 'misclassification', 'mail'),
        (2, 100, entropy, 'entropy', 'mail'),
        (2, 100, gini, 'gini', 'mail'),
        (2, 100, misclassification, 'misclassification', 'mail'),
        (3, 100, entropy, 'entropy', 'mail'),
        (3, 100, gini, 'gini', 'mail'),
        (3, 100, misclassification, 'misclassification', 'mail'),
        (1, 100, entropy, 'entropy', 'blob'),
        (1, 100, gini, 'gini', 'blob'),
        (1, 100, misclassification, 'misclassification', 'blob'),
        (2, 100, entropy, 'entropy', 'blob'),
        (2, 100, gini, 'gini', 'blob'),
        (2, 100, misclassification, 'misclassification', 'blob'),
        (3, 100, entropy, 'entropy', 'blob'),
        (3, 100, gini, 'gini', 'blob'),
        (3, 100, misclassification, 'misclassification', 'blob'),
    ]
    
    for params in params_list_0: # 0 or 1
        hyper_parameters = {
            'mdt': params[0], # Max depth of the tree
            'nt': 100, # Number of trees in the forest
            'h': params[2], # The evaluation function
            'hname': params[3],
            'dataset': params[4] # data sets, 0: blob, 1: spiral, 2: mail, 3: image
        }
        filename = 'project/Logs/ds-'+str(hyper_parameters['dataset'])+'_rf_nt-'+str(hyper_parameters['nt'])+'_mdt-'+str(hyper_parameters['mdt'])+'_h-'+str(hyper_parameters['hname'])+'.txt'
        
        print(f'Creating file {filename}')
        if run_analysis:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as log:
                log.write('num,train1,train2,train3,train4,train5,test1,test2,test3,test4,test5\n')
                rf = RandomForestModel(H=hyper_parameters['h'], 
                                    k=hyper_parameters['nt'],
                                    maxTreeDepth=hyper_parameters['mdt'])
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
                output = make_output_strings(dataset, rf, attributes)
                for line in output:
                    log.write(line)
                log.close()

    # visualize_log(filename)