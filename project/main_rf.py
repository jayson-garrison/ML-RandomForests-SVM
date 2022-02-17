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
    generate_visuals = False

    # Blobs only has two attributes, M=2 always
    params_blobs = [
        (1, entropy, 'entropy', 'blob', 2),
        (1, gini, 'gini', 'blob', 2),
        (1, misclassification, 'misclassification', 'blob', 2),
        (2, entropy, 'entropy', 'blob', 2),
        (2, gini, 'gini', 'blob', 2),
        (2, misclassification, 'misclassification', 'blob', 2),
        (3, entropy, 'entropy', 'blob', 2),
        (3, gini, 'gini', 'blob', 2),
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
    # Image has 3000 attributes, let M be in {2, 6, 11~log2(3000)}
    params_image = [
        (1, entropy, 'entropy', 'image', 2),
        (1, entropy, 'entropy', 'image', 6),
        (1, entropy, 'entropy', 'image', 11),
        (1, gini, 'gini', 'image', 2),
        (1, gini, 'gini', 'image', 6),
        (1, gini, 'gini', 'image', 11),
        (1, misclassification, 'misclassification', 'image', 2),
        (1, misclassification, 'misclassification', 'image', 6),
        (1, misclassification, 'misclassification', 'image', 11),
        (2, entropy, 'entropy', 'image', 2),
        (2, entropy, 'entropy', 'image', 6),
        (2, entropy, 'entropy', 'image', 11),
        (2, gini, 'gini', 'image', 2),
        (2, gini, 'gini', 'image', 6),
        (2, gini, 'gini', 'image', 11),
        (2, misclassification, 'misclassification', 'image', 2),
        (2, misclassification, 'misclassification', 'image', 6),
        (2, misclassification, 'misclassification', 'image', 11),
        (3, entropy, 'entropy', 'image', 2),
        (3, entropy, 'entropy', 'image', 6),
        (3, entropy, 'entropy', 'image', 11),
        (3, gini, 'gini', 'image', 2),
        (3, gini, 'gini', 'image', 6),
        (3, gini, 'gini', 'image', 11),
        (3, misclassification, 'misclassification', 'image', 2),
        (3, misclassification, 'misclassification', 'image', 6),
        (3, misclassification, 'misclassification', 'image', 11),
    ]
    # Mail has 256 attributes, let M be in {2, 4, 8}
    params_mail = [
        (1, entropy, 'entropy', 'mail', 2),
        (1, entropy, 'entropy', 'mail', 4),
        (1, entropy, 'entropy', 'mail', 8),
        (1, gini, 'gini', 'mail', 2),
        (1, gini, 'gini', 'mail', 4),
        (1, gini, 'gini', 'mail', 8),
        (1, misclassification, 'misclassification', 'mail', 2),
        (1, misclassification, 'misclassification', 'mail', 4),
        (1, misclassification, 'misclassification', 'mail', 8),
        (2, entropy, 'entropy', 'mail', 2),
        (2, entropy, 'entropy', 'mail', 4),
        (2, entropy, 'entropy', 'mail', 8),
        (2, gini, 'gini', 'mail', 2),
        (2, gini, 'gini', 'mail', 4),
        (2, gini, 'gini', 'mail', 8),
        (2, misclassification, 'misclassification', 'mail', 2),
        (2, misclassification, 'misclassification', 'mail', 4),
        (2, misclassification, 'misclassification', 'mail', 8),
        (3, entropy, 'entropy', 'mail', 2),
        (3, entropy, 'entropy', 'mail', 4),
        (3, entropy, 'entropy', 'mail', 8),
        (3, gini, 'gini', 'mail', 2),
        (3, gini, 'gini', 'mail', 4),
        (3, gini, 'gini', 'mail', 8),
        (3, misclassification, 'misclassification', 'mail', 2),
        (3, misclassification, 'misclassification', 'mail', 4),
        (3, misclassification, 'misclassification', 'mail', 8),
        
    ]
    
<<<<<<< HEAD
    for params in params_image:
=======
    for params in params_spiral:
>>>>>>> main
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
                output = make_output_strings(dataset, rf, attributes)
                for line in output:
                    log.write(line)
                log.close()

    if generate_visuals:
        log_dir = 'project/Logs/'
        for file in os.listdir(log_dir):
            visualize_log(file, file)
        # visualize_log(filename)