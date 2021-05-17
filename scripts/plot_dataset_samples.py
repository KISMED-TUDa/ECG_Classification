import matplotlib.pyplot as plt
import tikzplotlib

from wettbewerb import load_references


if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references("D:/ECG_Classification Data/training/")
    # Plot a few data samples
    for ecg_lead, ecg_label in zip(ecg_leads, ecg_labels):
        plt.plot(ecg_lead)
        tikzplotlib.save("plot.tex")
        pass