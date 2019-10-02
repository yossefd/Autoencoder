import keras
from keras.layers import Input, Dense, Activation, Reshape, Dropout, Layer
from keras.models import Model
import numpy as np
import os
import csv
import random
from keras.utils import multi_gpu_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import shutil
import math
from keras import objectives
import keras.backend as K
import jellyfish
from scipy import spatial
from keras import regularizers
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from sklearn.neighbors import KernelDensity
import seaborn as sns
import scipy.spatial.distance as dist
import scipy
import scipy.stats as stats
from keras.regularizers import l1
from sklearn import metrics
import tensorflow as tf
from sklearn.manifold import MDS
import matplotlib.lines as mlines
from mpl_toolkits import mplot3d


# check vadility of CDR3 sequence
def check_cdr3(seq):
    chars = ['#', 'X', '*', '_']
    if seq == '':
        return False
    for c in chars:
        if c in seq:
            return False
    return True


# find the tag of a sample according to the dataset, add datasets if needed
def tag_per_data_set(file_name, root):
    my_tag = 'error'
    if 'vaccine' in root:
        tmp = file_name.split('.')[0]
        my_tag = tmp.split('^')[1]
    if 'cancer' in root:
        tmp = file_name[0]
        if tmp == 'c':
            my_tag = 'Cancer'
        else:
            my_tag = 'Healthy'
    if 'benny_chain' in root:
        if 'naive' in file_name:
            my_tag = 'Naive'
        else:
            my_tag = 'Memory'
    if 'Glanville' in root:
        my_tag = file_name.split('.')[0]
    if 'Rudqvist' in root:
        my_tag = file_name.split('-')[0]
    if 'Sidhom' in root:
        if 'SIY' in file_name:
            my_tag = 'SIY'
        else:
            my_tag = 'TRP2'
    return my_tag


# load data by path - n data
def load_n_data(path, p, del_str, aa_str, v_str, f_str):
    all_data = []
    all_vs = []
    n_data = []
    n_tags = []
    n_files = []
    n_freqs = []
    n_vs = []
    for directory, subdirectories, files in os.walk(path):
        for i, file in enumerate(files):
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                vs = []
                freqs = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    data.append(cdr3)
                    if f_str != 'None':
                        freqs.append(row[f_str])
                    if v_str != 'None':
                        vs.append(row[v_str])
                all_data += data
                all_vs += vs

                # sample data by p
                if isinstance(p, float):
                    sample_n = int(len(data) * p)
                else:
                    sample_n = p
                inds = random.sample(list(range(len(data))), sample_n)
                n_data += [data[i] for i in inds]

                # freqs to counts
                if f_str != 'None':
                    n_freqs += [int(float(freqs[i])) for i in inds]
                if v_str != 'None':
                    n_vs += [vs[i] for i in inds]

                # tags
                t = tag_per_data_set(file, path)
                if t == 'error':
                    print('Warning: Add Data Set To (tag_per_data_set) Function !')
                    quit()
                n_tags += [t] * sample_n

                n_files += [file] * sample_n

    # check maximal length
    max_length = np.max([len(s) for s in all_data])
    return n_data, n_vs, n_tags, n_files, n_freqs, max_length, len(set(all_vs))


# load CDR3 sequences per sample
def load_seqs_by_samples(r, del_str, aa_str, n):
    seqs_dict = dict()
    for d, subd, files in os.walk(r):
        for file in files:
            with open(os.path.join(d, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    data.append(cdr3)
                if len(data) < n:
                    print(file + ': smaller then ' + str(n))
                    continue
                seqs_dict[file.split('.')[0]] = np.random.choice(data, n)
    return seqs_dict


# load V genes per sample
def load_vs_by_samples(r, del_str, v_str, n):
    all_vs = []
    vs_dict = dict()
    for d, subd, files in os.walk(r):
        for file in files:
            with open(os.path.join(d, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                vs = [row[v_str] for row in reader]
                if len(vs) < n:
                    print(file + ': smaller then ' + str(n))
                    continue
                vs = list(np.random.choice(vs, n))
                all_vs += vs
                vs_dict[file.split('.')[0]] = vs
    return vs_dict, set(all_vs)


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_data(path, n, del_str, aa_str, f_str):
    all_data = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                seqs = []
                freqs = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    seqs.append(cdr3)
                    if f_str != 'None':
                        freqs.append(int(float(row[f_str])))
                vecs = data_preprocessing(seqs, n)
                all_data[file.split('.')[0]] = {'vecs': vecs, 'seqs': seqs, 'freqs': freqs}
    return all_data


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_data_with_v(path, n, del_str, aa_str, v_str, vs_n, f_str):
    all_data = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                seqs = []
                vs = []
                freqs = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    seqs.append(cdr3)
                    if v_str != 'None':
                        vs.append(row[v_str])
                    if f_str != 'None':
                        freqs.append(int(float(row[f_str])))
                vecs = data_preprocessing_with_v(seqs, vs, n, vs_n)
                all_data[file.split('.')[0]] = {'vecs': vecs, 'seqs': seqs, 'freqs': freqs}
    return all_data


# cdr3s to one hot vectors
def data_preprocessing(string_set, max_length):
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


# cdr3s to one hot vectors
def data_preprocessing_with_v(string_set, vs_set, max_length, n_vs):
    # one hot vector per amino acid dictionary- wih STOP sequence- !
    aa = ['V', 'I', 'L', 'E', 'Q', 'D', 'N', 'H', 'W', 'F',
          'Y', 'R', 'K', 'S', 'T', 'M', 'A', 'G', 'P', 'C', '!']
    n_aa = len(aa)
    one_hot = {a: [0] * n_aa for a in aa}
    for key in one_hot:
        one_hot[key][aa.index(key)] = 1
    # add zero key for the zero padding
    one_hot['0'] = [0] * n_aa
    # v one-hot vectors
    all_vs = list(set(vs_set))
    one_hot_v = {v: [0] * n_vs for v in all_vs}
    for v in one_hot_v:
        one_hot_v[v][all_vs.index(v)] = 1
    # add 1 to the maximum length ( +1 for the ! stop signal)
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
        v += one_hot_v[vs_set[ind]]
        one_vecs.append(v)
    return one_vecs


# predictions to one-hot vectors
def hardmax_zero_padding(l):
    n = 21
    l_chunks = [l[i:i + n] for i in range(0, len(l), n)]
    l_new = []
    for chunk in l_chunks:
        new_chunk = list(np.zeros(n, dtype=int))
        max = np.argmax(chunk)
        # cut in the stop codon to remove zero padding
        if max == 20:
            break
        new_chunk[max] = 1
        l_new += new_chunk
    return l_new


# count mismatches between an input vector and the predicted vector
def count_mismatches_zero_padding(a, b):
    n = 21
    a_chunks = [a[i:i + n] for i in range(0, len(a), n)]
    b_chunks = [b[i:i + n] for i in range(0, len(b), n)]
    count_err = 0
    for ind, chunck_a in enumerate(a_chunks):
        ind_a = ''.join(str(x) for x in chunck_a).find('1')
        ind_b = ''.join(str(x) for x in b_chunks[ind]).find('1')
        if ind_a != ind_b:
            count_err += 1
        # early stopping when there are allready 2 mismatches
        if count_err > 2:
            return 3
    return count_err


# calculate accuracy, all vectors
def calc_accuracy_zero_padding(inputs, y, path):
    acc = 0
    acc1 = 0
    acc2 = 0
    n = len(inputs)
    for i in range(n):
        hard_max_y = hardmax_zero_padding(y[i])
        real = list(inputs[i])
        # cut the output to be the same length as input
        real = real[:len(hard_max_y)]
        if real == hard_max_y:
            acc += 1
            acc1 += 1
            acc2 += 1
        else:
            # accept 1 mismatch aa
            err = count_mismatches_zero_padding(real, hard_max_y)
            if err == 1:
                acc1 += 1
                acc2 += 1
            else:
                # accept 2 mismatch aa
                if err == 2:
                    acc2 += 1
    print('accuracy: ' + str(acc) + '/' + str(n) + ', ' + str(np.round((acc / n) * 100, 2)) + '%')
    print('1 mismatch accuracy: ' + str(acc1) + '/' + str(n) + ', ' + str(np.round((acc1 / n) * 100, 2)) + '%')
    print('2 mismatch accuracy: ' + str(acc2) + '/' + str(n) + ', ' + str(np.round((acc2 / n) * 100, 2)) + '%')
    with open(path + 'autoencoder_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(np.round((acc / n) * 100, 2)), '1 Mismatch': str(np.round((acc1 / n) * 100, 2)),
                         '2 Mismatch': str(np.round((acc2 / n) * 100, 2))})


# plot loss and accuracy
def plot_loss_acc(_accuracy, _val_accuracy, _loss, _val_loss, path):
    _epochs = range(len(_accuracy))
    plt.plot(_epochs, _accuracy, 'bo', label='Training accuracy')
    plt.plot(_epochs, _val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('classifier_accuracy_plot')
    plt.clf()
    plt.close()
    plt.figure()
    plt.plot(_epochs, _loss, 'bo', label='Training loss', color='mediumaquamarine')
    plt.plot(_epochs, _val_loss, 'b', label='Validation loss', color='cornflowerblue')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + 'loss_plot')
    plt.clf()
    plt.close()


# plot loss
def plot_loss(_loss, _val_loss, path):
    plt.figure()
    _epochs = range(len(_val_loss))
    plt.plot(_epochs, _loss, 'bo', label='Training loss', color='mediumaquamarine')
    plt.plot(_epochs, _val_loss, 'b', label='Validation loss', color='cornflowerblue')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + 'loss_plot')
    plt.clf()
    plt.close()


# calculate classifier accuracy
def calc_accuracy_classifier(true, pred, path):
    n = len(true)
    correct = 0.0
    for i, tag in enumerate(true):
        if tag == np.round(pred[i]):
            correct += 1
    print('classification accuracy:' + str(np.round(correct/n*100, 2)) + '%')
    with open(path + 'classifier_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(np.round(correct/n*100, 2))})


# calculate classifier accuracy by k vectors
def calc_accuracy_by_k_classifier(true, pred, n_arr, path):
    ns = []
    for n in n_arr:
        # dictionary: tag -> its predictions
        clusters = dict()
        for i, tag in enumerate(true):
            clusters[tag] = clusters.get(tag, []) + [pred[i]]
        # cluster to n vectors
        for tag in clusters:
            data = clusters[tag]
            random.shuffle(data)
            clusters[tag] = [data[x:x + n] for x in list(range(0, len(data), n))]
        # accuracy by clusters of n
        correct = 0.0
        for tag in clusters:
            for n_cluster in clusters[tag]:
                n_cluster = [np.round(i) for i in n_cluster]
                if n_cluster.count(tag) > len(n_cluster)/2:
                    correct += len(n_cluster)
        print(str(n) + ' classification accuracy:' + str(np.round(correct/len(true)*100, 2)) + '%')
        ns.append(str(np.round(correct/len(true)*100, 2)))
    with open(path + 'classifier_accuracy_clusters.csv', 'w') as csvfile:
        fieldnames = ['Accuracy ' + str(n) for n in n_arr]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy ' + str(n): ns[i] for i, n in enumerate(n_arr)})


# calculate classifier accuracy by whole sample
def calc_accuracy_by_sample(true, pred, files, k, path):
    samples = {}
    tags_files = {}
    for i, file in enumerate(files):
        samples[file] = samples.get(file, []) + [pred[i]]
        tags_files[file] = true[i]
    k = min([len(samples[file]) for file in samples])
    correct = 0
    for file in samples:
        k_preds = random.sample(samples[file], k)
        preds = [int(np.round(i)) for i in k_preds]
        if preds.count(1) > k/2:
            sample_tag = 1
        else:
            sample_tag = 0
        if sample_tag == tags_files[file]:
            correct += 1
    print('sample classification accuracy:' + str(np.round(correct/len(tags_files)*100, 2)) + '%')
    with open(path + 'classifier_accuracy_by_samples.csv', 'w') as csvfile:
        fieldnames = ['Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(np.round(correct/len(true)*100, 2))})


# calculate classifier weighted accuracy
def calc_weighted_accuracy(true, pred, files, fs, path):
    samples = {}
    tags_files = {}
    for i, file in enumerate(files):
        samples[file] = samples.get(file, []) + [[pred[i], fs[i]]]
        tags_files[file] = true[i]
    correct = 0
    for file in samples:
        weighted_score = np.sum([samples[file][i][0] * samples[file][i][1] for i in range(len(samples[file]))])
        sample_tag = int(np.round(weighted_score))
        if sample_tag == tags_files[file]:
            correct += 1
    print('sample weighted accuracy:' + str(np.round(correct/len(tags_files)*100, 2)) + '%')
    with open(path + 'classifier_weighted_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(np.round(correct/len(true)*100, 2))})


# load projections by sample
def load_representations(path, n):
    samples = dict()
    for file in os.listdir(path):
        if file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        if len(vecs_array) < n:
            n = len(vecs_array)
        vecs_array = random.sample([vec[0] for vec in vecs_array], n)
        samples[file] = np.array(vecs_array)
    return samples


# load projections by frequencies by sample
def load_representations_by_freqs(path, n):
    samples = dict()
    for file in os.listdir(path):
        if file.split('.')[1] != 'p':
            continue
        vecs_array = pickle.load(open(os.path.join(path, file), "rb"))
        if len(vecs_array) < n:
            n = len(vecs_array)
        vecs_array = random.sample([[vec[0], vec[2]] for vec in vecs_array], n)
        vecs_array_fs = []
        for v in vecs_array:
            vecs_array_fs += [list(v[0])] * v[1]
        samples[file] = np.array(vecs_array_fs)
    return samples


def scatter_tsne(data, inds, tag, neg_str, path, color1):
    color2 = 'lightsteelblue'
    for i, val in enumerate(data):
        if tag_per_data_set(inds[i], path) == tag:
            c = color1
            m = 'D'
            s = 7
        else:
            c = color2
            m = '.'
            s = 10
        plt.scatter(val[0], val[1], color=c, marker=m, s=s)
    plt.title('Autoencoder Projections TSNE')
    plt.tight_layout()
    patches = [mpatches.Patch(color=color1, label=tag),
               mpatches.Patch(color=color2, label=neg_str)]
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path)
    plt.clf()
    plt.close()


# kernel density distance
def kde_distance(set1, set2, flag=False):
    # calc distance matrix between both sets
    h = 1.06 * np.std(set1) * len(set1) ** (-1/5)
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(set2)
    dis = kde.score_samples(set1)
    # dont check density of point with itself when fit and score are on the same set-
    # remove constant K(0)/h from each point in the array
    if flag:
        zeros_set = [np.zeros(set1.shape[1])]
        kde_const = KernelDensity(bandwidth=h, kernel='gaussian')
        kde_const.fit(zeros_set)
        const = kde_const.score_samples(zeros_set)
        dis += const
    return np.average(np.exp(dis))


def plot_heatmap(data, diagonal, headers, s, round_s):
    sns.set(font_scale=0.5)
    for i, row in enumerate(data):
        for j in range(len(row)):
            if i != j:
                data[i][j] = data[i][j]/diagonal[i]

    sns.heatmap(data, xticklabels=headers, yticklabels=headers, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot_kws={"size": 10})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title(s + 'KDE Representations Distances')
    plt.tight_layout()
    plt.savefig(s + 'kde_representations_distances_' + round_s + '.png')
    plt.clf()
    plt.close()


def plot_self_bar(y_list, x_ticks, s, round_s):
    pos = np.arange(len(x_ticks))
    plt.bar(pos, [np.log(y) for y in y_list], align='center', alpha=0.5)
    plt.xticks(pos, x_ticks, rotation='vertical')
    plt.ylabel('KDE Distances')
    plt.title(s + 'KDE Within The Diagonal')
    plt.tight_layout()
    plt.savefig(s + 'kde_self_bar_' + round_s + '.png')
    plt.clf()
    plt.close()


# find publicity of each CDR3 sequence
def public_cdr3s(path, del_str, aa_str):
    seqs = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                for row in reader:
                    cdr3 = row[aa_str]
                    if not check_cdr3(cdr3):
                        continue
                    if cdr3 not in seqs:
                        seqs[cdr3] = set()
                    seqs[cdr3].add(file.split('.')[0])
    seqs = {seq: len(val) for seq, val in seqs.items()}
    return seqs


# load projections with sequences by sample
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


# find the center of all projections
def find_center(data):
    vecs = [v[0] for v in data]
    CM = np.average(vecs, axis=0)
    return CM


# find the all distanced from center
def distances_from_center(data, cm, public):
    distances = dict()
    vecs = [i[0] for i in data]
    dis_mat = dist.cdist(vecs, [cm])
    for j, p in enumerate(data):
        d = dis_mat[j][0]
        n = int(public[p[1]])
        distances[n] = distances.get(n, []) + [d]
    return distances


def dict_to_df(d, x_str, y_str):
    x = []
    y = []
    ordered_keys = np.sort(list(d.keys()))
    for key in ordered_keys:
        x += [key] * len(d[key])
        y += list(d[key])
    return pd.DataFrame(data={x_str: x, y_str: y})


def plot_distances(my_x, my_y, my_df, my_path, min=0, max=0):
    sns.boxplot(x=my_x, y=my_y, data=my_df, color="royalblue", boxprops=dict(alpha=.7))
    if min != 0 and max != 0:
        plt.ylim(min, max)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(my_path)
    plt.clf()
    plt.close()


# KDE within a sample
def self_density(d, public):
    densities = dict()
    for s in d:
        sample_vecs = [i[0] for i in d[s]]
        for i, val in enumerate(d[s]):
            kde = KernelDensity(bandwidth=3.0, kernel='gaussian')
            kde.fit(sample_vecs[:i] + sample_vecs[i+1:])
            dis = kde.score_samples([val[0]])
            density = np.exp(dis[0])
            n = int(public[val[1]])
            densities[n] = densities.get(n, []) + [density]
    return densities


# scatter projections by amino acid
def scatter_tsne_aa(data, seqs, aa, color, path):
    categories = {aa: color, 'Other': 'lightsteelblue'}
    for i, val in enumerate(data):
        if aa[0] in seqs[i][1:]:
            key = aa
            m = 'D'
            s = 7
        else:
            key = 'Other'
            m = '.'
            s = 10
        c = categories[key]
        plt.scatter(val[0], val[1], color=c, marker=m, s=s)
    plt.title('Projections TSNE ' + aa)
    plt.tight_layout()
    patches = []
    for key in categories:
        patches.append(mpatches.Patch(color=categories[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path + 'projections_tsne_' + aa)
    plt.clf()
    plt.close()


# scatter projections by lengths
def scatter_tsne_all_lengths(data, seqs, path):
    cs = ['dimgray', 'darkgray', 'lightgray',
          'lightcoral', 'coral', 'darkorange', 'orange', 'darkgoldenrod', 'olive', 'yellowgreen', 'lawngreen', 'lightgreen', 'mediumaquamarine',
          'c', 'cadetblue', 'skyblue', 'darkblue', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet', 'pink', 'crimson', 'r', 'brown', 'maroon']
    lens = list(set([len(i) for i in seqs]))
    cs_dict = {}
    for i in range(len(lens)):
        cs_dict[lens[i]] = cs[i]
    for i, val in enumerate(data):
        n = len(seqs[i])
        if n not in cs_dict:
            cs_dict[n] = cs.pop()
        c = cs_dict[n]
        plt.scatter(val[0], val[1], c=c, marker='.', s=10)
    plt.title('Projections TSNE , Lengths')
    plt.tight_layout()
    patches = []
    for key in np.sort(list(cs_dict.keys())):
        patches.append(mpatches.Patch(color=cs_dict[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path + 'projections_tsne_lengths')
    plt.clf()
    plt.close()


def scatter_cm(d, path):
    sns.boxplot(x='CDR3 Length', y='Distance From Center', data=d, color="crimson", boxprops=dict(alpha=.7))
    plt.title('Projections Radius By CDR3 Length')
    plt.tight_layout()
    plt.savefig(path + 'projections_radius_cdr3_length_boxplot')
    plt.clf()
    plt.close()


# t-test br plot per amino acid
def plot_t_bars(ts_, ps_, aa, color, path):
    sns.set(font_scale=0.8)
    pos = np.arange(len(ts_))
    labels = []
    for p in ps_:
        if p < 0.001:
            labels.append('***')
        else:
            if p < 0.01:
                labels.append('**')
            else:
                if p < 0.05:
                    labels.append('*')
                else:
                    labels.append('')
    plt.bar(pos, ts_, align='center', alpha=0.5, color=color)
    gaps = []
    for t in ts_:
        if t < 0:
            gaps.append(-1)
        else:
            gaps.append(0.1)
    for i in range(len(pos)):
        plt.text(i, ts_[i] + gaps[i], s=labels[i], ha='center', fontsize=8)
    plt.xticks(pos, pos, rotation='vertical')
    plt.ylabel('t Value')
    plt.title('Projections ' + aa + ' t-Test')
    plt.tight_layout()
    plt.savefig(path + 'projections_t-test_' + aa)
    plt.clf()
    plt.close()


# equal sampling of categories in train and test sets
def sample_equally(d, t):
    pos_inds = [i for i in range(len(t)) if t[i] == 1]
    zero_inds = [i for i in range(len(t)) if t[i] == 0]
    num_pos = len(pos_inds)
    num_neg = len(zero_inds)
    if num_pos > num_neg:
        pos_inds = random.sample(pos_inds, num_neg)
    else:
        zero_inds = random.sample(zero_inds, num_pos)
    d_tmp = [d[i] for i in pos_inds]
    d = d_tmp + [d[i] for i in zero_inds]
    t_tmp = [t[i] for i in pos_inds]
    t = t_tmp + [t[i] for i in zero_inds]
    return d, t


# custom sampling to train and test with frequencies
def my_train_test_split_freqs(vecs_data_, tags_, files_, freqs_, test_size=0.20):
    # equal sampling
    vecs_data_, tags_ = sample_equally(vecs_data_, tags_)
    # divide to train and test
    n = len(vecs_data_)
    k = int(n*test_size)
    all_inds = list(range(n))
    test_inds = list(random.sample(all_inds, k))
    train_inds = list(set(all_inds) - set(test_inds))
    train_x = [vecs_data_[i] for i in train_inds]
    test_x = [vecs_data_[i] for i in test_inds]
    train_y = [tags_[i] for i in train_inds]
    test_y = [tags_[i] for i in test_inds]
    test_files_ = [files_[i] for i in test_inds]
    test_freqs_ = [freqs_[i] for i in test_inds]
    return train_x, test_x, train_y, test_y, test_files_, test_freqs_


# custom sampling to train and test
def my_train_test_split(vecs_data_, tags_, files_, test_size=0.20):
    # equal sampling
    vecs_data_, tags_ = sample_equally(vecs_data_, tags_)
    # divide to trainand test
    n = len(vecs_data_)
    k = int(n*test_size)
    all_inds = list(range(n))
    test_inds = list(random.sample(all_inds, k))
    train_inds = list(set(all_inds) - set(test_inds))
    train_x = [vecs_data_[i] for i in train_inds]
    test_x = [vecs_data_[i] for i in test_inds]
    train_y = [tags_[i] for i in train_inds]
    test_y = [tags_[i] for i in test_inds]
    test_files_ = [files_[i] for i in test_inds]
    return train_x, test_x, train_y, test_y, test_files_


# categorical tags to 1 and 0
def tags_preprocessing(my_tag, all_tags):
    tags_vecs = []
    for t in all_tags:
        if t == my_tag:
            tags_vecs.append(1)
        else:
            tags_vecs.append(0)
    return tags_vecs


# calculate AUC
def calc_auc(y_list, pred_list):
    my_y = []
    my_pred = []
    for i, y in enumerate(y_list):
        my_y.append(y)
        my_pred.append(pred_list[i])
    fpr, tpr, thresholds = metrics.roc_curve(my_y, my_pred, pos_label=1)
    auc = np.round(metrics.auc(fpr, tpr), 4)
    return auc, fpr, tpr


# plot AUC
def plot_auc(d, path, root):
    for key in d:
        plt.plot(d[key]['fpr'], d[key]['tpr'], d[key]['color'], label=key + ' (area = ' + str(d[key]['auc']) + ')')
    plt.title(root + ' AUC')
    plt.legend(fontsize='x-small')
    plt.savefig(path + 'auc_plot.png')
    plt.clf()
    plt.close()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


# save KDE matrix to csv file
def kde_to_csv(dis_dict_, headers_, path_to_csv_):
    # fill the full matrix of the data
    with open(path_to_csv_, 'w') as csvfile:
        fieldnames = ['file'] + headers_
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, f in enumerate(headers_):
            data = {'file': f}
            for j, f2 in enumerate(headers_):
                # keep the real value in the csv file
                data[f2] = dis_dict_[f][f2]
            writer.writerow(data)


# read csv file to array
def csv_to_arr(path_to_csv_):
    headers_ = []
    data_arr_ = []
    self_arr_ = []
    with open(path_to_csv_, mode='r') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            del row['file']
            if i == 0:
                headers_ = list(row.keys())
            tmp = []
            for j, key in enumerate(row):
                val = np.float(row[key])
                if i == j:
                    self_arr_.append(val)
                    tmp.append(np.nan)
                else:
                    tmp.append(val)
            data_arr_.append(tmp)
    return data_arr_, self_arr_, headers_


# KDE distance between every two samples
def compute_kde_distances(vecs_dict_):
    dis_dict_ = dict()
    for i, file in enumerate(vecs_dict_):
        dis_dict_[file] = dict()
        for j, file2 in enumerate(vecs_dict_):
            # non-simetric distance metric
            if i == j:
                # keep seperate dict for self distances for bar plot
                dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2], flag=True)
            else:
                dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2])

            # for simetric distance metric
            # if i < j:
            #     dis_dict_[file][file2] = kde_distance(vecs_dict_[file], vecs_dict_[file2])
            # else:
            #     dis_dict_[file][file2] = dis_dict_[file2][file]

    return dis_dict_


# calculate V distribution of a sample
def calc_distribution(vs, vs_set):
    vs_dict = dict()
    # init with low values instead of 0 for the KL
    for v in vs_set:
        vs_dict[v] = 0.1
    # count segments
    for v in vs:
        vs_dict[v] += 1
    # find distribution
    n = len(vs)
    for v in vs_dict:
        vs_dict[v] = vs_dict[v]/n
    dist = []
    for v in vs_set:
        dist.append(vs_dict[v])
    return dist


# calculate KL between two distributions
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# plot KL distance heatmap
def kl_heatmap(data, headers, path):
    sns.set(font_scale=0.4)
    ax = sns.heatmap(data, xticklabels=headers, yticklabels=headers, cmap='coolwarm', annot_kws={"size": 3})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Vs distributions KL test')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()


# final distance between two samples
def similarity_min_avg(set1, set2, flag=False):
    d_list = []
    for ind1, seq1 in enumerate(set1):
        tmp = []
        for ind2, seq2 in enumerate(set2):
            if flag and ind1 == ind2:
                continue
            tmp.append(jellyfish.levenshtein_distance(seq1, seq2))
        d_list.append(min(tmp))
    return np.average(d_list)


# compute similarity between each pair of compartments using func
# for efficiency- only upper half will be computed and the results
# .. are saved for later lowe half
def cxc_similarity(d):
    data = []
    my_order = []
    seen = dict()
    for ind1, c1 in enumerate(d):
        my_order.append(c1)
        r = []
        for ind2, c2 in enumerate(d):
            sample1 = d[c1]
            sample2 = d[c2]
            if ind1 < ind2:
                # a measure of how similar both samples are according their cdr3 sets
                val = similarity_min_avg(sample1, sample2)
                seen[c2+'x'+c1] = val
            elif ind1 == ind2:
                # a measure of how similar both samples are according their cdr3 sets
                val = similarity_min_avg(sample1, sample2, flag=True)
                seen[c2+'x'+c1] = val
            else:
                val = seen[c1+'x'+c2]
            r.append(val)
        data.append(r)
    return data, my_order


# plot ED self-distances
def plot_diagonal(x, files, path):
    sns.set(font_scale=0.8)
    pos = np.arange(len(x))
    plt.bar(pos, x, align='center', alpha=0.5, color='b')
    plt.xticks(pos, files, rotation='vertical')
    plt.ylabel('ED Distance')
    plt.title('ED diagonal')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ed_distances_diagonal.png'))
    plt.clf()
    plt.close()


# heatmap plot of similarity between cdr3 sets of different compartments
def plot_similarity_mat(data_, comps, path):
    # nan at diagonal
    diag = []
    data = []
    for i in range(len(data_)):
        row = []
        for j in range(len(data_)):
            if i == j:
                diag.append(data_[i][j])
                val = np.nan
            else:
                val = data_[i][j]
            row.append(val)
        data.append(row)
    plot_diagonal(diag, comps, path)
    sns.set(font_scale=0.5)
    ax = sns.heatmap(data, xticklabels=comps, yticklabels=comps, cmap='coolwarm', annot_kws={"size": 3})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Min Average Edit Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ed_distances.png'))
    plt.clf()
    plt.close()


def scatter_3D_MDS(results, samples_list, my_path, s_round):
    colors = ['gold', 'm', 'blue', 'lime', 'orchid', 'grey', 'r']
    tags_set = set(samples_list)
    colors_dict = {t: colors[i] for i, t in enumerate(tags_set)}
    colors = [colors_dict[s] for s in samples_list]
    ax = plt.axes(projection='3d')
    for x, y, z, c in zip(results[:, 0], results[:, 1], results[:, 2], colors):
        ax.scatter3D([x], [y], [z], c=c, s=15, marker='x')
    patches = []
    for key in colors_dict:
        patches.append(mpatches.Patch(color=colors_dict[key], label=key))
    plt.legend(handles=patches, fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('KDE MDS')
    plt.xlabel('First MDS', fontsize=10)
    plt.ylabel('Second MDS', fontsize=10)
    plt.tight_layout()
    plt.savefig(my_path + s_round + '_MDS.png')
    plt.clf()
    plt.close()


class AutoEncoder:
    def __init__(self, input_set, weights_path, encoding_dim=3):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.num_classes = 2  # binary classifier
        self.weights_path = weights_path

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        decoded3 = Dense(self.input_shape, activation='elu')(dropout2)
        reshape = Reshape((int(self.input_shape/21), 21))(decoded3)
        decoded3 = Dense(21, activation='softmax')(reshape)
        reshape2 = Reshape(self.x[0].shape)(decoded3)
        model = Model(inputs, reshape2)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        self.autoencoder_model = model
        return model

    def _encoder_2(self):
        inputs = Input(shape=self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu', name='last_layer')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder_2 = model
        return model

    def fc(self, enco):
        fc1 = Dense(30,  activation='tanh')(enco)
        fc2 = Dense(15, activation='tanh')(fc1)
        dropout1 = Dropout(0.1)(fc2)
        fc3 = Dense(10, activation='tanh')(dropout1)
        dropout2 = Dropout(0.1)(fc3)
        out = Dense(1, activation='sigmoid')(dropout2)
        return out

    def classifier(self):
        ec = self._encoder_2()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        model = Model(inputs, self.fc(ec_out))
        self.classifier_model = model
        return model

    def fit_autoencoder(self, train_x, batch_size=10, epochs=300):
        self.autoencoder_model = multi_gpu_model(self.autoencoder_model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.autoencoder_model.compile(optimizer=adam, loss='mse')
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        results = self.autoencoder_model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                             epochs=epochs, batch_size=batch_size,
                                             callbacks=[tb_callback, es_callback])
        return results

    def fit_classifier(self, train_x, train_y, batch_size=10, epochs=300):
        self.classifier_model = multi_gpu_model(self.classifier_model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')

        # # train only fc
        # for layer in self.encoder_2.layers:
        #     layer.trainable = False
        # self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy',
        #                               metrics=['accuracy'])
        # results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
        #                           epochs=epochs, batch_size=batch_size,
        #                           callbacks=[tb_callback, es_callback])
        #
        # # train both encoder and fc
        # for layer in self.encoder_2.layers:
        #     layer.trainable = True
        self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy',
                                      metrics=['accuracy'])
        results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
                                            epochs=epochs, batch_size=batch_size,
                                            callbacks=[tb_callback])
        return results

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/decoder_weights.h5')
        self.autoencoder_model.save(r'./weights_' + self.weights_path + '/ae_weights.h5')

    def save_cl(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.classifier_model.save(r'./weights_' + self.weights_path + '/classifier_weights.h5')


class VAutoEncoder:
    def __init__(self, input_set, weights_path, vs_len, encoding_dim=3):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.vs_len = vs_len
        self.aa_shape = self.input_shape-self.vs_len
        self.lambda1 = self.create_lambda1()
        self.lambda2 = self.create_lambda2()
        self.weights_path = weights_path
        print(self.x)

    def create_lambda1(self):
        return self.MyLambda1(self.vs_len, self.aa_shape)

    def create_lambda2(self):
        return self.MyLambda2(self.vs_len)

    class MyLambda1(Layer):
        # this class is Lambda layer for the CDR3 input one hot split
        # exactly as writing instead: decoded3_0 = Lambda(lambda x: x[:, :-self.vs_len])(x)
        def __init__(self, vs_n, aa_n, **kwargs):
            self.vs_n = vs_n
            self.aa_n = aa_n
            super(VAutoEncoder.MyLambda1, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VAutoEncoder.MyLambda1, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, :-self.vs_n]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.aa_n)

        def get_config(self):
            config = super(VAutoEncoder.MyLambda1, self).get_config()
            config['vs_n'] = self.vs_n
            config['aa_n'] = self.aa_n
            return config

    class MyLambda2(Layer):
        # this class is Lambda layer for the Vs input one hot split
        # exactly as writing instead: decoded3_1 = Lambda(lambda x: x[:, -self.vs_len:])(x)
        def __init__(self, vs_n, **kwargs):
            self.vs_n = vs_n
            super(VAutoEncoder.MyLambda2, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VAutoEncoder.MyLambda2, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, -self.vs_n:]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.vs_n)

        def get_config(self):
            config = super(VAutoEncoder.MyLambda2, self).get_config()
            config['vs_n'] = self.vs_n
            return config

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        print(model.summary())
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        x = Dense(self.input_shape, activation='elu')(dropout2)
        # split the softmax layer for the 'aa one-hots' and 'v one-hot'
        decoded3_0 = self.lambda1(x)
        decoded3_1 = self.lambda2(x)
        reshape_0 = Reshape((int(self.aa_shape / 21), 21))(decoded3_0)
        reshape_1 = Reshape((1, self.vs_len))(decoded3_1)
        decoded4_0 = Dense(21, activation='softmax')(reshape_0)
        decoded4_1 = Dense(self.vs_len, activation='softmax')(reshape_1)
        reshape2_0 = Reshape((self.aa_shape, ))(decoded4_0)
        reshape2_1 = Reshape((self.vs_len, ))(decoded4_1)
        both = keras.layers.concatenate([reshape2_0, reshape2_1], axis=1)
        model = Model(inputs, both)
        print(model.summary())
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        self.autoencoder_model = model
        return model

    def fit_autoencoder(self, train_x, batch_size=10, epochs=300):
        self.autoencoder_model = multi_gpu_model(self.autoencoder_model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.autoencoder_model.compile(optimizer=adam, loss='mse')
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        results = self.autoencoder_model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                             epochs=epochs, batch_size=batch_size,
                                             callbacks=[tb_callback, es_callback])
        return results

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/v_encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/v_decoder_weights.h5')
        self.autoencoder_model.save(r'./weights_' + self.weights_path + '/v_ae_weights.h5')


class EmbeddingAutoEncoder:
    def __init__(self, input_set, D, weights_path, encoding_dim=3, batch_size=50, emb_alpha=0.1):
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.D = D
        self.weights_path = weights_path
        # embedding
        self.emb_alpha = emb_alpha
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        print(model.summary())
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        decoded3 = Dense(self.input_shape, activation='elu')(dropout2)
        reshape = Reshape((int(self.input_shape/21), 21))(decoded3)
        decoded3 = Dense(21, activation='softmax')(reshape)
        reshape2 = Reshape(self.x[0].shape)(decoded3)
        model = Model(inputs, reshape2)
        print(model.summary())

        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        # create the model using our input (cdr3 sequences set) and
        # two separate outputs -- one for the reconstruction of the
        # data and another for the representations, respectively
        model = Model(inputs=inputs, outputs=[dc_out, ec_out])
        self.model = model
        return model

    # 1. y_true 2. y_pred
    def vae_loss(self, D, ec_out):
        emb = 0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                if i < j:
                    # norm 2
                    dis_z = K.sqrt(K.sum(K.square(ec_out[i] - ec_out[j])))
                    emb += K.square(dis_z - D[i][j])
        emb = self.emb_alpha * emb
        return emb

    def generator(self, X, D, batch):
        # generate: A. batches of x B. distances matrix within batch
        while True:
            inds = []
            for i in range(batch):
                # choose random index in features
                index = np.random.choice(X.shape[0], 1)[0]
                if i == 0:
                    batch_X = np.array([X[index]])
                else:
                    batch_X = np.concatenate((batch_X, np.array([X[index]])), axis=0)
                inds.append(index)
            tmp = D[np.array(inds)]
            batch_D = tmp[:, np.array(inds)]
            # 1. training data-features 2. target data-labels
            yield batch_X, [batch_X, batch_D]

    def fit_generator(self, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=['mse', self.vae_loss])
        results = self.model.fit_generator(self.generator(self.x, self.D, self.batch_size),
                                           steps_per_epoch=self.x.shape[0]/self.batch_size,
                                           epochs=epochs, verbose=2)
        return results

    def fit_autoencoder(self, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=['mse', self.vae_loss], metrics=['mae'])
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        self.model.fit(x=self.x, y=[self.x, self.D], validation_split=0.2, verbose=2,
                       epochs=epochs, batch_size=self.batch_size,
                       callbacks=[tb_callback, es_callback])

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/embedding_encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/embedding_decoder_weights.h5')
        self.model.save(r'./weights_' + self.weights_path + '/embedding_ae_weights.h5')


class VEmbeddingAutoEncoder:
    def __init__(self, input_set, D, weights_path, vs_len, encoding_dim=3, batch_size=50, emb_alpha=0.1):
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.D = D
        self.vs_len = vs_len
        self.aa_shape = self.input_shape-self.vs_len
        self.lambda1 = self.create_lambda1()
        self.lambda2 = self.create_lambda2()
        self.weights_path = weights_path
        # embedding
        self.emb_alpha = emb_alpha
        print(self.x)

    def create_lambda1(self):
        return self.MyLambda1(self.vs_len, self.aa_shape)

    def create_lambda2(self):
        return self.MyLambda2(self.vs_len)

    class MyLambda1(Layer):
        # this class is Lambda layer for the CDR3 input one hot split
        # exactly as writing instead: decoded3_0 = Lambda(lambda x: x[:, :-self.vs_len])(x)
        def __init__(self, vs_n, aa_n, **kwargs):
            self.vs_n = vs_n
            self.aa_n = aa_n
            super(VEmbeddingAutoEncoder.MyLambda1, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VEmbeddingAutoEncoder.MyLambda1, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, :-self.vs_n]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.aa_n)

        def get_config(self):
            config = super(VEmbeddingAutoEncoder.MyLambda1, self).get_config()
            config['vs_n'] = self.vs_n
            config['aa_n'] = self.aa_n
            return config

    class MyLambda2(Layer):
        # this class is Lambda layer for the Vs input one hot split
        # exactly as writing instead: decoded3_1 = Lambda(lambda x: x[:, -self.vs_len:])(x)
        def __init__(self, vs_n, **kwargs):
            self.vs_n = vs_n
            super(VEmbeddingAutoEncoder.MyLambda2, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VEmbeddingAutoEncoder.MyLambda2, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, -self.vs_n:]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.vs_n)

        def get_config(self):
            config = super(VEmbeddingAutoEncoder.MyLambda2, self).get_config()
            config['vs_n'] = self.vs_n
            return config

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        print(model.summary())
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        x = Dense(self.input_shape, activation='elu')(dropout2)
        # split the softmax layer for the 'aa one-hots' and 'v one-hot'
        decoded3_0 = self.lambda1(x)
        decoded3_1 = self.lambda2(x)
        reshape_0 = Reshape((int(self.aa_shape / 21), 21))(decoded3_0)
        reshape_1 = Reshape((1, self.vs_len))(decoded3_1)
        decoded4_0 = Dense(21, activation='softmax')(reshape_0)
        decoded4_1 = Dense(self.vs_len, activation='softmax')(reshape_1)
        reshape2_0 = Reshape((self.aa_shape, ))(decoded4_0)
        reshape2_1 = Reshape((self.vs_len, ))(decoded4_1)
        both = keras.layers.concatenate([reshape2_0, reshape2_1], axis=1)
        # ---
        model = Model(inputs, both)
        print(model.summary())
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        # create the model using our input (cdr3 sequences set) and
        # two separate outputs -- one for the reconstruction of the
        # data and another for the representations, respectively
        model = Model(inputs=inputs, outputs=[dc_out, ec_out])
        self.model = model
        return model

    # 1. y_true 2. y_pred
    def vae_loss(self, D, ec_out):
        emb = 0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                if i < j:
                    # norm 2
                    dis_z = K.sqrt(K.sum(K.square(ec_out[i] - ec_out[j])))
                    # norm 1
                    # dis_z = K.sum(K.abs(ec_out[i] - ec_out[j]))
                    emb += K.square(dis_z - D[i][j])
        emb = self.emb_alpha * emb
        return emb

    def generator(self, X, D, batch):
        # generate: A. batches of x B. distances matrix within batch
        while True:
            inds = []
            for i in range(batch):
                # choose random index in features
                index = np.random.choice(X.shape[0], 1)[0]
                if i == 0:
                    batch_X = np.array([X[index]])
                else:
                    batch_X = np.concatenate((batch_X, np.array([X[index]])), axis=0)
                inds.append(index)
            tmp = D[np.array(inds)]
            batch_D = tmp[:, np.array(inds)]
            # 1. training data-features 2. target data-labels
            yield batch_X, [batch_X, batch_D]

    def fit_generator(self, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=['mse', self.vae_loss])
        results = self.model.fit_generator(self.generator(self.x, self.D, self.batch_size),
                                           steps_per_epoch=self.x.shape[0]/self.batch_size,
                                           epochs=epochs, verbose=2)
        return results

    def fit_autoencoder(self, train_x, batch_size=10, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss='mse')
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        results = self.model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                             epochs=epochs, batch_size=batch_size,
                                             callbacks=[tb_callback, es_callback])
        return results

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/v_embedding_encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/v_embedding_decoder_weights.h5')
        self.model.save(r'./weights_' + self.weights_path + '/v_embedding_ae_weights.h5')


class Classifier:
    def __init__(self, input_set, weights_path, encoding_dim=3, batch_size=50, emb_alpha=0.1):
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.weights_path = weights_path
        self.emb_alpha = emb_alpha  # embedding
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu', name='last_layer')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        return model

    def _fc(self, enco):
        fc1 = Dense(30,  activation='tanh')(enco)
        fc2 = Dense(15, activation='tanh')(fc1)
        dropout1 = Dropout(0.1)(fc2)
        fc3 = Dense(10, activation='tanh')(dropout1)
        dropout2 = Dropout(0.1)(fc3)
        out = Dense(1, activation='sigmoid')(dropout2)
        return out

    def classifier(self):
        ec = self._encoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        model = Model(inputs, self._fc(ec_out))
        self.classifier_model = model
        return model

    def fit_classifier(self, train_x, train_y, batch_size=10, epochs=300):
        # self.classifier_model = multi_gpu_model(self.classifier_model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')

        # # train only fc
        # for layer in self.encoder.layers:
        #     layer.trainable = False
        # self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy',
        #                               metrics=['accuracy'])
        # results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
        #                           epochs=epochs, batch_size=batch_size,
        #                           callbacks=[tb_callback, es_callback])

        # # train both encoder and fc
        # for layer in self.encoder_2.layers:
        #     layer.trainable = True
        self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2, epochs=epochs, batch_size=batch_size,
                                            callbacks=[tb_callback, es_callback])
        return results

    def save_cl(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.classifier_model.save(r'./weights_' + self.weights_path + '/classifier_weights.h5')


if __name__ == '__main__':

    # Load Data ----------------------------------------------

    print('loading data')

    root = 'benny_chain_processed_data'  # path to data set
    save_path = root + '_Results'

    # replace the headers with those in the 'csv' file, put 'None' if missing
    DEL_S = ','  # which delimiter to use to separate headers
    CDR3_S = 'amino_acid'  # header of amino acid sequence of CDR3
    V_S = 'None'  # header of V gene, 'None' for data set with no resolved V
    F_S = 'total_reads'  # header of clone frequency, 'None' for data set with no frequency

    EPOCHS = 100  # number of epochs for each model
    P_LOAD = 1.0  # number of data to load for training, float or int
    ENCODING_DIM = 30  # number of dimensions in the embedded space
    N_PROJECTIONS = 100  # number of projections to use in further anlysis
    N_FOR_DISTANCES = 100  # number of projections to use in the ED and KL

    # each directory contains the projections for each sample and all the results
    encoder_dir = os.path.join(save_path, 'encoder_projections/')
    v_encoder_dir = os.path.join(save_path, 'v_encoder_projections/')
    embedding_dir = os.path.join(save_path, 'embedding_projections/')
    v_embedding_dir = os.path.join(save_path, 'v_embedding_projections/')

    # load data
    data, vs, all_tags, files, freqs, MAX_LEN, V_ONE_HOT_LEN = load_n_data(root, P_LOAD, DEL_S, CDR3_S, V_S, F_S)
    vecs_data = data_preprocessing(data, MAX_LEN)  # representation of the sequences as one-hot vectors
    # if the dataset contains V, another representation of the sequences as one-hot vectors with V
    if V_S != 'None':
        vecs_data_vs = data_preprocessing_with_v(data, vs, MAX_LEN, V_ONE_HOT_LEN)

    # Run Parameters- change the analysis you wish to run to True, otherwise- False
    RUN_AE = False
    RUN_V_AE = False
    RUN_EMB = False
    RUN_V_EMB = False
    RUN_CLS_ENC = False
    RUN_CLS_ENC_V = False
    RUN_CLS_EMB = False
    RUN_CLS_EMB_V = False
    RUN_SAVE = False
    RUN_TSNE = False
    RUN_KDE = True
    RUN_PROPS = False
    RUN_KL = False
    RUN_ED = False
    RUN_MDS = False

    # one color for each category within the dataset, add if necessary
    colors = ['gold', 'm', 'blue', 'lime', 'orchid', 'grey', 'r']

    # Auto-encoder ----------------------------------------------
    # run the basic auto-encoder

    if RUN_AE:

        print('auto-encoder')

        # train + test sets
        train_X, test_X, tmp1, tmp2 = train_test_split(vecs_data, vecs_data, test_size=0.20)

        # train  auto-encoder model
        ae = AutoEncoder(np.array(train_X), root, encoding_dim=ENCODING_DIM)
        ae.encoder_decoder()
        train_results = ae.fit_autoencoder(np.array(train_X), batch_size=50, epochs=EPOCHS)
        ae.save_ae()

        create_dir(encoder_dir)

        # plot loss and accuracy as a function of time
        plot_loss(train_results.history['loss'], train_results.history['val_loss'], encoder_dir)

        # load trained model
        encoder = load_model(r'./weights_' + root + '/encoder_weights.h5')
        decoder = load_model(r'./weights_' + root + '/decoder_weights.h5')
        test_X = np.array(test_X)

        # auto-encoder predictions
        x = encoder.predict(test_X)
        y = decoder.predict(x)

        # accuracy
        calc_accuracy_zero_padding(test_X, y, encoder_dir)

    # Auto-encoder With V ----------------------------------------------
    # run the auto-encoder with V

    if RUN_V_AE and V_S != 'None':

        print('v auto-encoder')

        # train + test sets
        train_X, test_X, tmp1, tmp2 = train_test_split(vecs_data_vs, vecs_data_vs, test_size=0.20)

        # train  auto-encoder model
        v_ae = VAutoEncoder(np.array(train_X), root, V_ONE_HOT_LEN, encoding_dim=ENCODING_DIM)
        v_ae.encoder_decoder()
        train_results = v_ae.fit_autoencoder(np.array(train_X), batch_size=50, epochs=EPOCHS)
        v_ae.save_ae()

        create_dir(v_encoder_dir)

        # plot loss and accuracy as a function of time
        plot_loss(train_results.history['loss'], train_results.history['val_loss'], v_encoder_dir)

        # load trained model
        custom_objects = {'MyLambda1': VAutoEncoder.MyLambda1, 'MyLambda2': VAutoEncoder.MyLambda2}
        encoder = load_model(r'./weights_' + root + '/v_encoder_weights.h5')
        decoder = load_model(r'./weights_' + root + '/v_decoder_weights.h5', custom_objects=custom_objects)
        test_X = np.array(test_X)

        # auto-encoder predictions
        x = encoder.predict(test_X)
        y = decoder.predict(x)

        # accuracy
        calc_accuracy_zero_padding(test_X, y, v_encoder_dir)

    # Embedding Auto-encoder ----------------------------------------------
    # run the auto-encoder with respect to the original distances

    if RUN_EMB:

        print('embedding')

        # train + test sets
        train_X, test_X, train_y, test_y = train_test_split(vecs_data, vecs_data, test_size=0.2)
        train_X = np.array(train_X)

        # calculate D- original distances matrix- norm 2 between all pairs of one-hot vectors
        D = spatial.distance.cdist(train_X, train_X, 'euclidean')

        # train model
        emb_ae = EmbeddingAutoEncoder(train_X, D, root, encoding_dim=ENCODING_DIM, batch_size=50, emb_alpha=0.01)
        emb_ae.encoder_decoder()
        embedding_train = emb_ae.fit_generator(epochs=EPOCHS)
        emb_ae.save_ae()

        create_dir(embedding_dir)

        # test model
        encoder = load_model(r'./weights_' + root + '/embedding_encoder_weights.h5')
        decoder = load_model(r'./weights_' + root + '/embedding_decoder_weights.h5')
        test_X = np.array(test_X)

        x = encoder.predict(test_X)
        y = decoder.predict(x)

        # accuracy
        calc_accuracy_zero_padding(test_X, y, embedding_dir)

    # Embedding Auto-encoder With V ----------------------------------------------
    # run the auto-encoder with respect to the original distances + V representation

    if RUN_V_EMB and V_S != 'None':

        print('v embedding')

        # train + test sets
        train_X, test_X, train_y, test_y = train_test_split(vecs_data_vs, vecs_data_vs, test_size=0.2)
        train_X = np.array(train_X)

        # calculate D- input distances matrix- norm 2
        D = spatial.distance.cdist(train_X, train_X, 'euclidean')

        # train model
        v_emb_ae = VEmbeddingAutoEncoder(train_X, D, root, V_ONE_HOT_LEN, encoding_dim=ENCODING_DIM, batch_size=50, emb_alpha=0.01)
        v_emb_ae.encoder_decoder()
        embedding_train = v_emb_ae.fit_generator(epochs=EPOCHS)
        v_emb_ae.save_ae()

        create_dir(v_embedding_dir)

        # test model
        custom_objects = {'MyLambda1': VEmbeddingAutoEncoder.MyLambda1, 'MyLambda2': VEmbeddingAutoEncoder.MyLambda2}
        encoder = load_model(r'./weights_' + root + '/v_embedding_encoder_weights.h5')
        decoder = load_model(r'./weights_' + root + '/v_embedding_decoder_weights.h5', custom_objects=custom_objects)
        test_X = np.array(test_X)

        x = encoder.predict(test_X)
        y = decoder.predict(x)

        # accuracy
        calc_accuracy_zero_padding(test_X, y, v_embedding_dir)

    #  Classifier ----------------------------------------------
    # Run classifier for each of the trained models

    # all possible runs
    classifier_runs = []
    if RUN_CLS_ENC:
        classifier_runs.append((encoder_dir, 'encoder_weights', vecs_data))
    if RUN_CLS_EMB:
        classifier_runs.append((embedding_dir, 'embedding_encoder_weights', vecs_data))
    if RUN_CLS_ENC_V and V_S != 'None':
        classifier_runs.append((v_encoder_dir, 'v_encoder_weights', vecs_data_vs))
    if RUN_CLS_EMB_V and V_S != 'None':
        classifier_runs.append((v_embedding_dir, 'v_embedding_encoder_weights', vecs_data_vs))

    for run_dir, run_weights, run_vecs in classifier_runs:

        results_dict = dict()

        # one vs. all classifier
        classes = set(all_tags)
        for j, tag in enumerate(classes):

            print('----------------')
            print(tag)

            # categorical to numerical tags
            tags = tags_preprocessing(tag, all_tags)

            # split to train and tests sets
            if F_S == 'None':
                train_X, test_X, train_Y, test_Y, test_files = my_train_test_split(run_vecs, tags, files, test_size=0.20)
            else:
                train_X, test_X, train_Y, test_Y, test_files, test_freqs = my_train_test_split_freqs(run_vecs, tags, files, freqs, test_size=0.20)
            train_X = np.array(train_X)

            cl = Classifier(train_X, root, encoding_dim=ENCODING_DIM, batch_size=10, emb_alpha=0.01)
            cl.classifier()
            # use the auto-encoder layers in the classifier
            model_ae = load_model(r'./weights_' + root + '/' + run_weights + '.h5')
            cl.encoder.set_weights(model_ae.get_weights())
            classify_train = cl.fit_classifier(np.array(train_X), np.array(train_Y), batch_size=10, epochs=EPOCHS)
            cl.save_cl()

            # plot loss and accuracy as a function of time
            accuracy = classify_train.history['acc']
            val_accuracy = classify_train.history['val_acc']
            loss = classify_train.history['loss']
            val_loss = classify_train.history['val_loss']
            plot_loss_acc(accuracy, val_accuracy, loss, val_loss, run_dir)

            # load model
            classifier = load_model(r'./weights_' + root + '/classifier_weights.h5')
            test_X = np.array(test_X)
            test_Y = np.array(test_Y)

            # classifier predictions
            preds = classifier.predict(test_X)
            preds = [i[0] for i in preds]

            # calculate accuracies
            calc_accuracy_classifier(test_Y, preds, run_dir)  # regular accuracy
            calc_accuracy_by_k_classifier(test_Y, preds, [5, 10, 20], run_dir)  # major vote out of groups of k accuracy
            calc_accuracy_by_sample(test_Y, preds, test_files, 5, run_dir)  # major vote by whole sample accuracy
            if F_S != 'None':
                calc_weighted_accuracy(test_Y, preds, test_files, test_freqs, run_dir)  # weighted accuracy

            # auc
            auc_, fpr_, tpr_ = calc_auc(test_Y, preds)

            results_dict[tag] = {'auc': auc_, 'fpr': fpr_, 'tpr': tpr_, 'color': colors[j]}
            print('AUC = ' + str(auc_))

            K.clear_session()

        plot_auc(results_dict, run_dir, root)

    # Save Projections ----------------------------------------------
    # save all projections per sample for each of the trained models

    if RUN_SAVE:

        print('save projections')

        # load data (without V) and models
        samples = load_all_data(root, MAX_LEN, DEL_S, CDR3_S, F_S)
        encoder = load_model(r'./weights_' + root + '/encoder_weights.h5')
        embedding_encoder = load_model(r'./weights_' + root + '/embedding_encoder_weights.h5')

        # save all projections per sample
        for s in samples:

            # encoder projections
            x_encoder = encoder.predict(np.array(samples[s]['vecs']))
            if F_S != 'None':
                info = [[x_encoder[i], samples[s]['seqs'][i], samples[s]['freqs'][i]] for i in range(len(x_encoder))]
            else:
                info = [[x_encoder[i], samples[s]['seqs'][i]] for i in range(len(x_encoder))]
            pickle.dump(info, open(encoder_dir + str(s) + '.p', "wb"))

            # encoder+distances projections
            x_embedding = embedding_encoder.predict(np.array(samples[s]['vecs']))
            if F_S != 'None':
                info = [[x_embedding[i], samples[s]['seqs'][i], samples[s]['freqs'][i]] for i in range(len(x_embedding))]
            else:
                info = [[x_embedding[i], samples[s]['seqs'][i]] for i in range(len(x_embedding))]
            pickle.dump(info, open(embedding_dir + str(s) + '.p', "wb"))

        if V_S != 'None':

            # load data with V and models
            samples_with_v = load_all_data_with_v(root, MAX_LEN, DEL_S, CDR3_S, V_S, V_ONE_HOT_LEN, F_S)
            v_encoder = load_model(r'./weights_' + root + '/v_encoder_weights.h5')
            v_embedding_encoder = load_model(r'./weights_' + root + '/v_embedding_encoder_weights.h5')

            # save all projections per sample
            for s in samples_with_v:

                # encoder projections
                x_v_encoder = v_encoder.predict(np.array(samples_with_v[s]['vecs']))
                if F_S != 'None':
                    info = [[x_v_encoder[i], samples_with_v[s]['seqs'][i], samples_with_v[s]['freqs'][i]] for i in range(len(x_v_encoder))]
                else:
                    info = [[x_v_encoder[i], samples_with_v[s]['seqs'][i]] for i in range(len(x_v_encoder))]
                pickle.dump(info, open(v_encoder_dir + str(s) + '.p', "wb"))

                # encoder+distances projections
                x_v_embedding_encoder = v_embedding_encoder.predict(np.array(samples_with_v[s]['vecs']))
                if F_S != 'None':
                    info = [[x_v_embedding_encoder[i], samples_with_v[s]['seqs'][i], samples_with_v[s]['freqs'][i]] for i in
                            range(len(x_v_embedding_encoder))]
                else:
                    info = [[x_v_embedding_encoder[i], samples_with_v[s]['seqs'][i]] for i in range(len(x_v_embedding_encoder))]
                pickle.dump(info, open(v_embedding_dir + str(s) + '.p', "wb"))

    # TSNE ----------------------------------------------
    # visualize all samples using TSNE

    if RUN_TSNE:

        print('TSNE')

        # projections to use
        path = encoder_dir

        # load projections
        vecs_dict = load_representations(path, N_PROJECTIONS)

        # calculte TSNE
        inds = []
        for i, f in enumerate(vecs_dict):
            if i == 0:
                X = vecs_dict[f]
            else:
                X = np.concatenate((X, vecs_dict[f]), axis=0)
            inds += [f] * len(vecs_dict[f])
        T = TSNE(n_components=2).fit_transform(X)

        # scatter TSNE
        classes = set(all_tags)
        for j, tag in enumerate(classes):
            scatter_tsne(T, inds, tag, 'Other', os.path.join(path, 'tsne_results_' + tag), colors[j])

    # KDE ----------------------------------------------
    # run KDE to find all pairwise distances between samples
    # run for each of the trained models twice, with and without frequencies

    if RUN_KDE:

        print('KDE')

        for path in [encoder_dir, v_encoder_dir, embedding_dir, v_embedding_dir]:

            if path == v_encoder_dir and V_S == 'None':
                continue

            # load projections
            vecs_dict = load_representations(path, N_PROJECTIONS)
            if F_S == 'None':
                vecs_dict_fs = 'None'
            else:
                vecs_dict_fs = load_representations_by_freqs(path, N_PROJECTIONS)

            for round in [[vecs_dict, 'no_freqs'], [vecs_dict_fs, 'with_freqs']]:

                if round[0] == 'None':
                    continue

                # caclculate KDE for every pair of samples
                dis_dict = compute_kde_distances(round[0])

                # save distances in csv
                headers = list(dis_dict.keys())
                path_to_csv = path + 'kde_representations_' + round[1] + '.csv'
                kde_to_csv(dis_dict, headers, path_to_csv)
                data_arr, self_arr, headers = csv_to_arr(path_to_csv)

                # plot distances, heatmap without diagonal and bar plot of self-distances
                plot_heatmap(data_arr, self_arr, headers, path, round[1])
                plot_self_bar(self_arr, headers, path, round[1])

    # Projections Properties ----------------------------------------------
    # check repertoire features within the embedded space

    if RUN_PROPS:

        print('projections properties')

        # Public Clones

        path = encoder_dir  # projections to use
        public_cdr3_dict = public_cdr3s(root, DEL_S, CDR3_S)  # public cdr3 sequences
        # load projections with their original sequences
        projections_dict = load_representations_and_seqs(path, N_PROJECTIONS)
        projections_arr = []
        for s in projections_dict:
            projections_arr += projections_dict[s]
        center_p = find_center(projections_arr)  # center of all projections in the dataset
        public_distances = distances_from_center(projections_arr, center_p, public_cdr3_dict)  # distance from center

        # plot distances results
        x = 'Publicity'
        y = 'Distance from center'
        distances_df = dict_to_df(public_distances, x, y)
        plot_distances(x, y, distances_df, path + 'public_distance_from_center.png')
        public_densities = self_density(projections_dict, public_cdr3_dict)
        x = 'Publicity'
        y = 'Self-Density'
        densities_df = dict_to_df(public_densities, x, y)
        plot_distances(x, y, densities_df, path + 'public_self_density.png')

        # Sequence Features (AA, Length)

        path = encoder_dir  # projections to use
        # load projections with their original sequences
        projections_dict = load_representations_and_seqs(path, N_PROJECTIONS)
        # split to vectors and sequences
        vecs = []
        seqs = []
        for s in projections_dict:
            vecs += [v[0] for v in projections_dict[s]]
            seqs += [v[1] for v in projections_dict[s]]

        # calculate TSNE
        T = TSNE(n_components=2).fit_transform(np.array(vecs))

        # projections by amino-acids
        n_pos = len(vecs[0])
        cs = ['magenta', 'springgreen', 'cornflowerblue']
        for i, a in enumerate(['Cysteine', 'Proline', 'Glycine']):
            scatter_tsne_aa(T, seqs, a, cs[i], path)
            check_list = []
            for seq in seqs:
                if a[0] in seq[1:]:
                    check_list.append(1)
                else:
                    check_list.append(0)
            ts = []
            ps = []
            for ind in range(n_pos):
                v = []
                x = []
                for j, vec in enumerate(vecs):
                    if check_list[j] == 1:
                        v.append(vec[ind])
                    else:
                        x.append(vec[ind])
                t, p = scipy.stats.ttest_ind(x, v, equal_var=False)
                ts.append(t)
                ps.append(p)
            plot_t_bars(ts, ps, a, cs[i], path)

        # projections by lengths
        scatter_tsne_all_lengths(T, seqs, path)

        # distance from center
        CM = np.average(vecs, axis=0)
        lens = []
        dis = []
        for i in range(len(vecs)):
            lens.append(len(seqs[i]))
            dis.append(np.linalg.norm(vecs[i] - CM))
        df = pd.DataFrame(data={'CDR3 Length': lens, 'Distance From Center': dis})
        scatter_cm(df, path)

    # KL Distances ----------------------------------------------

    if RUN_KL and V_S != 'None':

        print('KL')

        # load all Vs
        vs_samples, all_vs = load_vs_by_samples(root, DEL_S, V_S, N_FOR_DISTANCES)

        # calculate distributions
        dist_dict = dict()
        for file in vs_samples:
            dist_dict[file] = calc_distribution(vs_samples[file], all_vs)

        # compute KL between every two samples
        kl_dict = dict()
        for i, file in enumerate(dist_dict.keys()):
            kl_dict[file] = dict()
            for j, file2 in enumerate(dist_dict.keys()):
                if i == j:
                    kl_dict[file][file2] = np.nan
                elif i < j:
                    kl_dict[file][file2] = KL(dist_dict[file], dist_dict[file2])
                else:
                    kl_dict[file][file2] = kl_dict[file2][file]

        # fill the full matrix of the data and save csv
        with open(os.path.join(save_path, 'kl_distances.csv'), 'w') as csvfile:
            fieldnames = ['file'] + list(kl_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            data_arr = []
            for f in kl_dict:
                data = {'file': f}
                tmp = []
                for f2 in kl_dict:
                    val = kl_dict[f][f2]
                    data[f2] = val
                    tmp.append(val)
                writer.writerow(data)
                data_arr.append(tmp)

        # plot distances
        kl_heatmap(data_arr, list(kl_dict.keys()), os.path.join(save_path, 'kl_distances.png'))

    # ED Distances ----------------------------------------------

    if RUN_ED:

        print('ED')

        # load all sequences
        cdr3_samples = load_seqs_by_samples(root, DEL_S, CDR3_S, N_FOR_DISTANCES)

        # average min edit distance (levenshtein distance)
        all_ds, files_list = cxc_similarity(cdr3_samples)

        # plot distances
        plot_similarity_mat(all_ds, files_list, save_path)

        # fill the full matrix of the data and save to csv
        with open(os.path.join(save_path, 'ed_distances.csv'), 'w') as csvfile:
            fieldnames = ['file'] + files_list
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, val in enumerate(all_ds):
                file = files_list[i]
                data = {'file': file}
                for j, file2 in enumerate(fieldnames[1:]):
                    data[file2] = val[j]
                writer.writerow(data)

    # MDS ----------------------------------------------
    # run MDS for each KDE matrix

    if RUN_MDS:

        print('MDS')

        for path in [encoder_dir, v_encoder_dir, embedding_dir]:

            if path == v_encoder_dir and V_S == 'None':
                continue

            flag = True
            if F_S == 'None':
                flag = False

            for round in ['no_freqs', 'with_freqs']:

                if round == 'with_freqs' and not flag:
                    continue

                path_to_csv = path + 'kde_representations_' + round + '.csv'
                data_arr, self_arr, headers = csv_to_arr(path_to_csv)
                headers = [tag_per_data_set(file, root) for file in headers]

                # find minimum to remove baseline
                check_min = []
                for i in range(len(data_arr)):
                    for j in range(len(data_arr)):
                        if i == j:
                            data_arr[i][j] = 0.0
                        else:
                            check_min.append(data_arr[i][j])

                baseline = np.array(check_min).min()
                for i in range(len(data_arr)):
                    for j in range(len(data_arr)):
                        if i != j:
                            data_arr[i][j] -= baseline
                data_arr = np.array(data_arr)

                # calculate MDS
                model = MDS(n_components=3, random_state=1)
                out = model.fit_transform(data_arr)

                # scatter MDS
                scatter_3D_MDS(out, headers, path, round)









