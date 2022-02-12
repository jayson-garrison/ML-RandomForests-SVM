from RandomForest.RandomForestModel import *
from Utils.email_loader import load_email_data
from Utils.five_fold import five_fold

if __name__ == "__main__":
    dataset, attributes = load_email_data()
    partition = five_fold(dataset)
    # for idx in range(len(partition)):
    #     training = list()
    #     testing = partition[i]
    rf = RandomForestModel(maxTreeDepth=3)
    rf.fit(partition[0][0], attributes, 100)
    num, den = 0, 0
    for sample in partition[0][1]:
        prediction = rf.call(sample)
        if prediction == sample.getLabel(): num += 1
        den += 1
    print(f'Testing accuracy: {num/den}')
    num, den = 0, 0
    for sample in partition[0][0]:
        prediction = rf.call(sample)
        if prediction == sample.getLabel(): num += 1
        den += 1
    print(f'Training accuracy: {num/den}')