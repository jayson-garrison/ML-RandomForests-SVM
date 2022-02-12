from RandomForest.RandomForestModel import *
from Utils.email_loader import load_email_data
from project.Utils.five_fold import five_fold

if __name__ == "__main__":
    dataset, attributes = load_email_data()
    partition = five_fold(dataset)

    # for idx in range(len(partition)):
    #     training = list()
    #     testing = partition[i]
    rf = RandomForestModel(maxTreeDepth=3)
    rf.fit(partition[0], attributes, 5)
    num, den = 0, 0
    for sample in partition[1]:
        prediction = rf.call(sample)
        if prediction == sample.getLabel(): num += 1
        den += 1
    print(f'Validation accuracy: {num/den}')
