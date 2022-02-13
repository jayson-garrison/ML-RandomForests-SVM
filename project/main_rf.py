from fileinput import filename
import os
from RandomForest.RandomForestModel import *
from Utils.email_loader import load_email_data
from Utils.artificial_data_loader import *
from Utils.five_fold import *
from Utils.visualize_log import *


if __name__ == "__main__":
    hyper_parameters = {
        'mdt': 1, # Max depth of the tree
        'nt': 100, # Number of trees in the forest
        'h': entropy, # The evaluation function
        'hname': 'entropy'
    }
    filename = 'project/Logs/rf_nt-'+str(hyper_parameters['nt'])+'_mdt-'+str(hyper_parameters['mdt'])+'_h-'+str(hyper_parameters['hname'])+'.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as log:
        log.write('num,train1,train2,train3,train4,train5,test1,test2,test3,test4,test5\n')
        rf = RandomForestModel(H=hyper_parameters['h'], 
                            k=hyper_parameters['nt'],
                            maxTreeDepth=hyper_parameters['mdt'])
        dataset, attributes = load_artificial_data('./project/Datasets/artificial_datasets/dataset/blobs.csv')
        output = make_output_strings(dataset, rf, attributes)
        for line in output:
            log.write(line)
        log.close()
    visualize_log(filename)