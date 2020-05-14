import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import jellyfish
import pickle
import random
import shutil


def check_cdr3(seq):
    chars = ['#', 'X', '*', '_']
    if seq == '':
        return False
    for c in chars:
        if c in seq:
            return False
    return True


# load data by path - n data, directory or one file
def load_n_data(path, p, del_str, aa_str):
    samples_data = {}
    for directory, subdirectories, files in os.walk(path):
        for i, file in enumerate(files):
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    data.append(cdr3)
                if isinstance(p, float):
                    sample_n = int(len(data) * p)
                else:
                    sample_n = p
                inds = random.sample(list(range(len(data))), sample_n)
                samples_data[file.split('.')[0]] = [data[i] for i in inds]
    return samples_data


# compute edit-distance between every seq from set1 and every seq in set2
def comps_distances(set1, set2):
    dis_dict = dict()
    for seq1 in set1:
        for seq2 in set2:
            d = jellyfish.levenshtein_distance(seq1, seq2)  # edit distance
            dis_dict[d] = dis_dict.get(d, 0) + 1
    return dis_dict


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def plot_dis_hist(dis_dict, path, f1, f2):
    keys_sorted = [key for (key, value) in sorted(dis_dict.items())]
    vals_sorted = [value for (key, value) in sorted(dis_dict.items())]
    plt.bar(keys_sorted, vals_sorted, color='cornflowerblue')
    plt.xlabel('Count  ', fontsize=13)
    plt.ylabel('Number Of Mismatches ', fontsize=13)
    plt.xticks(keys_sorted, fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Distances Histogram")
    plt.tight_layout()
    plt.savefig(path + f1 + '_vs_' + f2 + '_comps_distance_hist')


if __name__ == '__main__':

    root = 'Glanville'
    save_path = root + '_Results'

    CDR3_S = 'aminoAcid'
    DEL_S = '\t'

    P_LOAD = 1.0

    results_dir = 'mismatches_distribution_glanville/'
    create_dir(results_dir)

    data = load_n_data(root, P_LOAD, DEL_S, CDR3_S)

    ind = 0

    for i, f1 in enumerate(data):
        for j, f2 in enumerate(data):

            if i == j:
                continue

            if ind == 3:
                break

            distances = comps_distances(data[f1], data[f2])
            plot_dis_hist(distances, results_dir, f1, f2)

            ind += 1
