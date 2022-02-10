from Utils.email_loader import load_email_data

if __name__ == "__main__":
    dataset = load_email_data()
    print(dataset[0])