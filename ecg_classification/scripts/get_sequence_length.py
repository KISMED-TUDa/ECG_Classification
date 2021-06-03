from wettbewerb import load_references

if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="/Users/Vladi/Desktop/training/")

    # Get max and min sequence length
    print("Max length =", max([ecg_lead.shape[0] for ecg_lead in ecg_leads]))
    print("Min length =", min([ecg_lead.shape[0] for ecg_lead in ecg_leads]))

