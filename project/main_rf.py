from Utils.email_loader import load_email_data
from RandomForest.RandomForestModel import *

if __name__ == "__main__":
    dataset = load_email_data()
    rf = RandomForestModel()
    print(rf.plurality_value(dataset))