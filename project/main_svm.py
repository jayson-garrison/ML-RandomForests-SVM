from SupportVectorMachine.SVM import *
from Utils.email_loader import load_email_data
from Utils.image_loader import load_image_data
from Utils.artificial_data_loader import *
from Utils.five_fold import *
from Utils.visualize_log import *
import os


params_blobs = [
    # params of the form (dataset name, softness constant C, tolerance tol, kernel function name)
    ('blobs', 10, .01, 'inner_product'),
    ('blobs', .1, .1, 'gaussian'),
    ('blobs', .1, .1, 'mystery'),
    ('blobs', .1, .001, 'inner_product'),
    ('blobs', .1, .001, 'gaussian'),
    ('blobs', .1, .001, 'mystery'),
    ('blobs', 10, .1, 'inner_product'),
    ('blobs', 10, .1, 'gaussian'),
    ('blobs', 10, .1, 'mystery'),
    ('blobs', 10, .001, 'inner_product'),
    ('blobs', 10, .001, 'gaussian'),
    ('blobs', 10, .001, 'mystery'),
    ('blobs', 1000, .1, 'inner_product'),
    ('blobs', 1000, .1, 'gaussian'),
    ('blobs', 1000, .1, 'mystery'),
    ('blobs', 1000, .001, 'inner_product'),
    ('blobs', 1000, .001, 'gaussian'),
    ('blobs', 1000, .001, 'mystery'),
]

params_spiral = [
    # params of the form (dataset name, softness constant C, tolerance tol, kernel function name)
    ('sprial', .1, .1, 'inner_product'),
    ('sprial', .1, .1, 'gaussian'),
    ('sprial', .1, .1, 'mystery'),
    ('sprial', .1, .001, 'inner_product'),
    ('sprial', .1, .001, 'gaussian'),
    ('sprial', .1, .001, 'mystery'),
    ('sprial', 10, .1, 'inner_product'),
    ('sprial', 10, .1, 'gaussian'),
    ('sprial', 10, .1, 'mystery'),
    ('sprial', 10, .001, 'inner_product'),
    ('sprial', 10, .001, 'gaussian'),
    ('sprial', 10, .001, 'mystery'),
    ('sprial', 1000, .1, 'inner_product'),
    ('sprial', 1000, .1, 'gaussian'),
    ('sprial', 1000, .1, 'mystery'),
    ('sprial', 1000, .001, 'inner_product'),
    ('sprial', 1000, .001, 'gaussian'),
    ('sprial', 1000, .001, 'mystery'),
]

params_email = [
    # params of the form (dataset name, softness constant C, tolerance tol, kernel function name)
    ('email', .1, .1, 'inner_product'),
    ('email', .1, .1, 'gaussian'),
    ('email', .1, .1, 'mystery'),
    ('email', .1, .001, 'inner_product'),
    ('email', .1, .001, 'gaussian'),
    ('email', .1, .001, 'mystery'),
    ('email', 10, .1, 'inner_product'),
    ('email', 10, .1, 'gaussian'),
    ('email', 10, .1, 'mystery'),
    ('email', 10, .001, 'inner_product'),
    ('email', 10, .001, 'gaussian'),
    ('email', 10, .001, 'mystery'),
    ('email', 1000, .1, 'inner_product'),
    ('email', 1000, .1, 'gaussian'),
    ('email', 1000, .1, 'mystery'),
    ('email', 1000, .001, 'inner_product'),
    ('email', 1000, .001, 'gaussian'),
    ('email', 1000, .001, 'mystery'),
]


if __name__ == "__main__":
    run_analysis = True
    generate_visuals = False

    for params in params_blobs:
        # params of the form (dataset name, softness constant C, tolerance tol, kernel function name)
        # params = ('blobs', 10, .01, 'inner_product') # for testing
        hyper_parameters = {
            'dataset': params[0], 
            'C': params[1],
            'tol': params[2],
            'kernel': params[3]
        }

        filename = 'project/Logs/SVM/DataSet_'+str(hyper_parameters['dataset'])+'/'+\
                                '_Softness-'+str(hyper_parameters['C'])+\
                                '_Tolerance-'+str(hyper_parameters['tol'])+\
                                '_Kernel-'+str(hyper_parameters['kernel'])+'.txt'
        
        print(f'Creating file {filename}')
        if run_analysis:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as log:
                log.write('train1,train2,train3,train4,train5,test1,test2,test3,test4,test5\n')

                if hyper_parameters['dataset'] == 'blobs':
                    dataset, attributes, X, Y = load_artificial_data('./project/Datasets/artificial_datasets/dataset/blobs.csv', using_svm=True)
                elif hyper_parameters['dataset'] == 'spiral':
                    dataset, attributes, X, Y = load_artificial_data('./project/Datasets/artificial_datasets/dataset/spirals.csv', using_svm=True)
                elif hyper_parameters['dataset'] == 'email':
                    dataset, attributes, X, Y = load_email_data(using_svm=True)
                else:
                    print('No data set given..')
                    exit()

                # RECALL: __init__(self, X, Y, C, tol, max_passes, k='inner_product')
                svm = SVM(X=X, 
                          Y=Y, 
                          C=hyper_parameters['C'], 
                          tol=hyper_parameters['tol'], 
                          max_passes=3, 
                          k=hyper_parameters['kernel'])

                output = make_output_strings_svm(dataset, svm) # had svm as second arg
                for line in output:
                    log.write(line)
                log.close()
                exit()
