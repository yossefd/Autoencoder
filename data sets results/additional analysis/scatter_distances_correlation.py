import os
import pickle
import random
from scipy import spatial
import numpy as np
import scipy.stats as stats
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shutil


def data_preprocessing(string_set):
    # one hot vector per amino acid dictionary- wih STOP sequence- !
    aa = ['V', 'I', 'L', 'E', 'Q', 'D', 'N', 'H', 'W', 'F',
          'Y', 'R', 'K', 'S', 'T', 'M', 'A', 'G', 'P', 'C', '!']
    n_aa = len(aa)
    one_hot = {a: [0] * n_aa for a in aa}
    for key in one_hot:
        one_hot[key][aa.index(key)] = 1
    # add zero key for the zero padding
    one_hot['0'] = [0] * n_aa
    # add 1 to the maximum length ( +1 for the ! stop signal)
    max_length = np.max([len(seq) for seq in string_set])
    max_length += 1
    # generate one-hot long vector for each cdr3
    one_vecs = []
    for ind, cdr3 in enumerate(string_set):
        # add stop signal in each sequence
        cdr3 = cdr3 + '!'
        my_len = len(cdr3)
        # zero padding in the end of the sequence
        if my_len < max_length:
            add = max_length - my_len
            cdr3 = cdr3 + '0'*add
        # one hot vectors
        v = []
        for c in cdr3:
            v += one_hot[c]
        one_vecs.append(v)
    return one_vecs


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def load_representations_and_seqs(path, n):
    samples = dict()
    for file in os.listdir(path):
        if file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        if len(vecs_array) < n:
            n = len(vecs_array)
        vecs_array = random.sample([[vec[0], vec[1]] for vec in vecs_array], n)
        samples[file] = vecs_array
    return samples


def scatter_distance(x, y, path, f):
    plt.scatter(x, y, color='black', marker="d", s=1)
    plt.title('CDR3 One-Hots vs. Autoencoder representations Distances', fontsize=13)
    plt.xlabel('Real  ', fontsize=13)
    plt.ylabel('Predicted ', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    r, pval = stats.stats.spearmanr(x, y)
    patchR = mpatches.Patch(label='r = ' + str(round(r, 6)), color='None')
    patchP = mpatches.Patch(label='p = ' + str(round(pval, 6)), color='None')
    plt.legend(handles=[patchR, patchP], fontsize=10)
    plt.tight_layout()
    plt.savefig(path + f + '_real_vs_predicted_distances.png')
    plt.clf()
    plt.close()


if __name__ == '__main__':

    root = 'vaccine_TCR_processed_data'
    save_path = root + '_Results'

    encoder_dir = os.path.join(save_path, 'encoder_projections/')
    path = encoder_dir
    results_dir = 'encoder_distances_correlation_vaccine/'

    # embedding_dir = os.path.join(save_path, 'embedding_projections/')
    # path = embedding_dir
    # results_dir = 'embedding_distances_correlation_vaccine/'

    create_dir(results_dir)

    N_PROJECTIONS = 100

    projections_dict = load_representations_and_seqs(path, N_PROJECTIONS)

    for ind, file in enumerate(projections_dict):
        vecs = np.array([i[0] for i in projections_dict[file]])
        cdr3s = [i[1] for i in projections_dict[file]]
        cdr3s = np.array(data_preprocessing(cdr3s))
        # norm 2 distance
        D_cdr3 = spatial.distance.cdist(cdr3s, cdr3s, 'euclidean')
        D_reps = spatial.distance.cdist(vecs, vecs, 'euclidean')
        cdr3s_dis = []
        vecs_dis = []
        for i in range(len(cdr3s)):
            for j in range(i):
                cdr3s_dis.append(D_cdr3[j][i])
                vecs_dis.append(D_reps[j][i])
        scatter_distance(cdr3s_dis, vecs_dis, results_dir, file)
        if ind > 3:
            break
