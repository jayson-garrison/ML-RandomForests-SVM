from Utils.email_loader import load_email_data
from RandomForest.RandomForestModel import *

if __name__ == "__main__":
    dataset, attributes = load_email_data()
    rf = RandomForestModel(maxTreeDepth=3)
    initial_parents = set()
    tree = rf.learn_decision_tree(dataset, attributes, initial_parents)
    tree.pretty_print_tree()