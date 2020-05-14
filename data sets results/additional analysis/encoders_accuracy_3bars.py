import os
import csv
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path


# TWO BARS PLOT
def bar_plot(d, title, path):
    names = list(d.keys())
    fig, ax = plt.subplots()
    xlim = ax.get_xlim()
    N = len(names)
    ind = np.arange(N)  # X LOCATIONS FOR THE GROUP
    ind = [i * (xlim[1] / N) for i in ind]
    width = (ind[1] - ind[0]) * 0.2
    keys = tuple(d[names[0]].keys())
    rects1 = ax.bar(ind, [d[i][keys[0]] for i in d], width, color='r')
    rects2 = ax.bar(ind + width, [d[i][keys[1]] for i in d], width, color='darksalmon')
    rects3 = ax.bar(ind + 2*width, [d[i][keys[2]] for i in d], width, color='cornflowerblue')
    ax.set_title(title, fontsize=13)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(names)
    ax.legend((rects1[0], rects2[0], rects3[0]), ['0 Mismatch', '1 Mismatch', '2 Mismatch'], fontsize=10)
    plt.xticks(rotation=90, fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    fig.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()


def all_accuracies(data_sets, dir, title, save_path):
        acc = {}
        for root in data_sets:
            path = os.path.join(root + '_Results', dir)
            my_file = Path(os.path.join(path, 'autoencoder_accuracy.csv'))
            if my_file.exists():
                with open(os.path.join(path, 'autoencoder_accuracy.csv'), mode='r') as infile:
                        reader = csv.DictReader(infile, delimiter=',')
                        for row in reader:
                            acc[data_sets[root]] = {key: float(val) for key, val in row.items()}
            else:
                print(path)
        bar_plot(acc, title, save_path)


if __name__ == '__main__':

    data_sets = {'vaccine_TCR_processed_data': 'Vaccine',
                 'Sidhom': 'Sidhom',
                 'Rudqvist': 'Rudqvist',
                 'cancer data IEDB': 'Cancer',
                 'Glanville': 'Glanville',
                 'benny_chain_processed_data': 'Naive_Memory'}

    all_accuracies(data_sets, 'encoder_projections',
                   'Encoder Accuracy All Data Sets', 'encoders_acc_3bars.png')
    all_accuracies(data_sets, 'embedding_projections',
                   'Embedding Accuracy All Data Sets', 'embedding_encoders_acc_3bars.png')
    all_accuracies(data_sets, 'v_encoder_projections',
                   'Encoder With V Accuracy All Data Sets', 'v_encoders_acc_3bars.png')
    all_accuracies(data_sets, 'v_embedding_projections',
                   'Embedding With V  Accuracy All Data Sets', 'v_embedding_encoders_acc_3bars.png')
