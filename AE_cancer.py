import keras
from keras.layers import Input, Dense, Activation, Reshape, Dropout
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
from keras.regularizers import l2
from sklearn import metrics



# load data by path - n data, directory or one file
def load_n_data(path, p, del_str, aa_str, tag_str, file_check_ind):
    all_data = []
    all_data_n = []
    all_tags = []
    all_files = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                data = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if cdr3 == '':
                        continue
                    if '#' in cdr3:
                        continue
                    if 'X' in cdr3:
                        continue
                    if '*' in cdr3:
                        continue
                    data.append(cdr3)
                all_data += data

                # change --------
                # sample_n = int(len(data) * p)
                # all_data_n += random.sample(data, sample_n)
                sample_n = 1000
                if len(data) < sample_n:
                    all_data_n += data
                else:
                    all_data_n += random.sample(data, sample_n)
                # change --------

                # tags
                if file_check_ind == 0:
                    check_file = file[0]
                else:
                    check_file = file
                if tag_str in check_file:
                    t = 1
                else:
                    t = 0
                all_tags += [t] * sample_n
                all_files += [file] * sample_n
    # check maximal length
    max_length = np.max([len(s) for s in all_data])
    return all_data_n, all_tags, all_files, max_length


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_data(path, n, del_str, aa_str, v_str, j_str, total_str=0):
    all_data = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                seqs = []
                ns = []
                vs = []
                js = []
                for row in reader:
                    cdr3 = row[aa_str]
                    if cdr3 == '':
                        continue
                    if '#' in cdr3:
                        continue
                    if 'X' in cdr3:
                        continue
                    if '*' in cdr3:
                        continue
                    seqs.append(cdr3)
                    if total_str != 0:
                        ns.append(row[total_str])
                    vs.append(row[v_str])
                    js.append(row[j_str])
                vecs = data_preprocessing(seqs, n)
                all_data[file.split('.')[0]] = {'vecs': vecs, 'seqs': seqs, 'ns': ns, 'vs': vs, 'js': js}
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
    for cdr3 in string_set:
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


def hardmax_zero_padding(l):
    n = 21
    l_chunks = [l[i:i + n] for i in range(0, len(l), n)]
    l_new = []
    for chunk in l_chunks:
        new_chunk = list(np.zeros(n, dtype=int))
        # # taking the max only in place where not everything is 0
        # if not all(v == 0 for v in chunk):
        max = np.argmax(chunk)
        if max == 20:
            break
        new_chunk[max] = 1
        l_new += new_chunk
    return l_new


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
    print('accuracy: ' + str(acc) + '/' + str(n) + ', ' + str(round((acc / n) * 100, 2)) + '%')
    print('1 mismatch accuracy: ' + str(acc1) + '/' + str(n) + ', ' + str(round((acc1 / n) * 100, 2)) + '%')
    print('2 mismatch accuracy: ' + str(acc2) + '/' + str(n) + ', ' + str(round((acc2 / n) * 100, 2)) + '%')
    with open(path + 'autoencoder_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(round((acc / n) * 100, 2)), '1 Mismatch': str(round((acc1 / n) * 100, 2)),
                         '2 Mismatch': str(round((acc2 / n) * 100, 2))})


def plot_loss_acc(_epochs, _accuracy, _val_accuracy, _loss, _val_loss, path):
    plt.plot(_epochs, _accuracy, 'bo', label='Training accuracy')
    plt.plot(_epochs, _val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('classifier_accuracy_plot')
    plt.figure()
    plt.plot(_epochs, _loss, 'bo', label='Training loss', color='mediumaquamarine')
    plt.plot(_epochs, _val_loss, 'b', label='Validation loss', color='cornflowerblue')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + '_loss_plot')
    plt.clf()


def plot_loss(_epochs,_loss, _val_loss, path):
    plt.figure()
    plt.plot(_epochs, _loss, 'bo', label='Training loss', color='mediumaquamarine')
    plt.plot(_epochs, _val_loss, 'b', label='Validation loss', color='cornflowerblue')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + '_loss_plot')
    plt.clf()


def calc_accuracy_classifier(true, pred, path):
    n = len(true)
    correct = 0.0
    for i, tag in enumerate(true):
        if tag == round(pred[i]):
            correct += 1
    print('classification accuracy:' + str(round(correct/n*100, 2)) + '%')
    with open(path + 'classifier_accuracy.csv', 'w') as csvfile:
        fieldnames = ['Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(round(correct/n*100, 2))})


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
                n_cluster = [round(i) for i in n_cluster]
                if n_cluster.count(tag) > len(n_cluster)/2:
                    correct += len(n_cluster)
        print(str(n) + ' classification accuracy:' + str(round(correct/len(true)*100, 2)) + '%')
        ns.append(str(round(correct/len(true)*100, 2)))
    with open(path + 'classifier_accuracy_clusters.csv', 'w') as csvfile:
        fieldnames = ['Accuracy ' + str(n) for n in n_arr]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy ' + str(n): ns[i] for i, n in enumerate(n_arr)})


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
        preds = [int(round(i)) for i in k_preds]
        if preds.count(1) > k/2:
            sample_tag = 1
        else:
            sample_tag = 0
        if sample_tag == tags_files[file]:
            correct += 1
    print('sample classification accuracy:' + str(round(correct/len(tags_files)*100, 2)) + '%')
    with open(path + 'classifier_accuracy_by_samples.csv', 'w') as csvfile:
        fieldnames = ['Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(round(correct/len(tags_files)*100, 2))})


def load_representations(path):
    n = 400
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


def scatter_tsne(data, inds, check_tag, check_ind, pos_str, neg_str, path):
    color1 = 'salmon'
    color2 = 'magenta'
    for i, val in enumerate(data):
        if check_ind == 0:
            if inds[i][0] == check_tag:
                c = color1
            else:
                c = color2
        else:
            if check_tag in inds[i]:
                c = color1
            else:
                c = color2
        plt.scatter(val[0], val[1], color=c, marker='.', s=10)
    plt.title('Autoencoder Projections TSNE')
    plt.tight_layout()
    patches = [mpatches.Patch(color=color1, label=pos_str),
               mpatches.Patch(color=color2, label=neg_str)]
    plt.legend(handles=patches, fontsize='small', loc=2)
    plt.savefig(path)


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


def plot_heatmap(data, diagonal, headers, s):
    sns.set(font_scale=0.5)
    for i, row in enumerate(data):
        for j in range(len(row)):
            if i != j:
                data[i][j] = data[i][j]/diagonal[i]

    sns.heatmap(data, xticklabels=headers, yticklabels=headers, cmap='coolwarm', annot_kws={"size": 10})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title(s + 'KDE Representations Distances')
    plt.tight_layout()
    plt.savefig(s + 'kde_representations_distances.png')
    plt.clf()
    plt.close()


def plot_self_bar(y_list, x_ticks, s):
    pos = np.arange(len(x_ticks))
    plt.bar(pos, [np.log(y) for y in y_list], align='center', alpha=0.5)
    plt.xticks(pos, x_ticks, rotation='vertical')
    # plt.ylim(bottom=1.4*1e-71)
    plt.ylabel('KDE Distances')
    plt.title(s + 'KDE Within The Diagonal')
    plt.tight_layout()
    plt.savefig(s + 'kde_self_bar.png')


def public_cdr3s(path, del_str, aa_str):
    seqs = {}
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.DictReader(infile, delimiter=del_str)
                for row in reader:
                    cdr3 = row[aa_str]
                    if cdr3 == '':
                        continue
                    if '#' in cdr3:
                        continue
                    if 'X' in cdr3:
                        continue
                    if '*' in cdr3:
                        continue
                    if cdr3 not in seqs:
                        seqs[cdr3] = set()
                    seqs[cdr3].add(file.split('.')[0])
    seqs = {seq: len(val) for seq, val in seqs.items()}
    return seqs


def load_representations_and_seqs(path):
    n = 400
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


def find_center(data):
    vecs = [v[0] for v in data]
    CM = np.average(vecs, axis=0)
    return CM


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
    # sns.set(style="white", palette="muted")
    # sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.2})
    sns.boxplot(x=my_x, y=my_y, data=my_df, color="royalblue", boxprops=dict(alpha=.7))
    if min != 0 and max != 0:
        plt.ylim(min, max)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(my_path)
    plt.clf()
    plt.close()


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
    # sns.set(style="white", palette="muted")
    # sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.2})
    sns.boxplot(x='CDR3 Length', y='Distance From Center', data=d, color="crimson", boxprops=dict(alpha=.7))
    plt.title('Projections Radius By CDR3 Length')
    plt.tight_layout()
    plt.savefig(path + 'projections_radius_cdr3_length_boxplot')
    plt.clf()
    plt.close()


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


def my_train_test_split(vecs_data, tags, files, test_size=0.20):
    n = len(vecs_data)
    k = int(n*test_size)
    all_inds = list(range(n))
    test_inds = list(random.sample(all_inds, k))
    train_inds = list(set(all_inds) - set(test_inds))
    train_x = [vecs_data[i] for i in train_inds]
    test_x = [vecs_data[i] for i in test_inds]
    train_y = [tags[i] for i in train_inds]
    test_y = [tags[i] for i in test_inds]
    return train_x, test_x, train_y, test_y, [files[i] for i in test_inds]


def calc_auc(y_list, pred_list):
    my_y = []
    my_pred = []
    for i, y in enumerate(y_list):
        my_y.append(y)
        my_pred.append(pred_list[i])
    fpr, tpr, thresholds = metrics.roc_curve(my_y, my_pred, pos_label=1)
    auc = np.round(metrics.auc(fpr, tpr), 4)
    return auc, fpr, tpr


def plot_auc(d, root, path):
    plt.plot(d['fpr'], d['tpr'], 'm', label=' (area = ' + str(d['auc']) + ')')
    plt.title(root + ' AUC')
    plt.legend(fontsize='x-small')
    plt.savefig(path + 'auc_plot.png')


class AutoEncoder:
    def __init__(self, input_set, weights_path, encoding_dim=3):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.num_classes = 2  # binary classifier
        self.weights_path = weights_path
        # print(self.x)

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
        # print(model.summary())
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
        # print(model.summary())
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        # print(model.summary())
        self.autoencoder_model = model
        return model

    def _encoder_2(self):
        inputs = Input(shape=self.x[0].shape)
        # print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu', name='last_layer')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder_2 = model
        # print(model.summary())
        return model

    def fc(self, enco):
        # first try ---
        # fc2 = Dense(15, activation='elu', kernel_regularizer=l2(0.1))(enco)
        # dropout1 = Dropout(0.5)(fc2)
        # out = Dense(1, activation='sigmoid')(dropout1)

        # fc1 = Dense(30, activation='elu')(enco)
        # fc2 = Dense(15, activation='elu')(fc1)
        # dropout1 = Dropout(0.5)(fc2)
        fc3 = Dense(10, activation='elu')(enco)
        # dropout2 = Dropout(0.5)(fc3)
        out = Dense(1, activation='sigmoid')(fc3)
        return out

    def classifier(self):
        ec = self._encoder_2()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        model = Model(inputs, self.fc(ec_out))
        # print(model.summary())
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
                                                    patience=10, verbose=0, mode='auto')
        results = self.autoencoder_model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                             epochs=epochs, batch_size=batch_size,
                                             callbacks=[tb_callback, es_callback])
        return results

    def fit_classifier(self, train_x, train_y, batch_size=10, epochs=300):
        self.classifier_model = multi_gpu_model(self.classifier_model, gpus=3)
        # adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        sgd = keras.optimizers.SGD(lr=0.0833, momentum=0.9, decay=1e-2, nesterov=False)
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=10, verbose=0, mode='auto')

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
        self.classifier_model.compile(optimizer=sgd, loss='binary_crossentropy',
                                      metrics=['accuracy'])
        results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
                                            epochs=epochs, batch_size=batch_size,
                                            callbacks=[tb_callback, es_callback])
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


class Embedding_AutoEncoder:
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
        # encoded3 = Dense(self.encoding_dim, activation='elu', activity_regularizer=regularizers.l1(10e-5))(dropout2)
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

    def fit(self, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=3)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=['mse', self.vae_loss], metrics=['mae'])
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=10, verbose=0, mode='auto')
        self.model.fit(x=self.x, y=[self.x, self.D], validation_split=0.2, verbose=2,
                       epochs=epochs, batch_size=self.batch_size,
                       callbacks=[tb_callback, es_callback])

    def save(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/embedding_encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/embedding_decoder_weights.h5')
        self.model.save(r'./weights_' + self.weights_path + '/embedding_ae_weights.h5')


if __name__ == '__main__':

    # Load Data ----------------------------------------------

    root = 'cancer data IEDB'

    # load samples with all lengths
    data, tags, files, max_len = load_n_data(root, 1, ',', 'Chain 2 CDR3 Curated', 'c', 0)

    vecs_data = data_preprocessing(data, max_len)

    # Auto-encoder ----------------------------------------------

    # train + test sets
    train_X, test_X, tmp1, tmp2 = train_test_split(vecs_data, vecs_data, test_size=0.20)

    # train  auto-encoder model
    ae = AutoEncoder(np.array(train_X), root, encoding_dim=30)
    ae.encoder_decoder()
    autoencoder_train = ae.fit_autoencoder(np.array(train_X), batch_size=50, epochs=500)  # 300
    ae.save_ae()

    encoder_dir = root + '_encoder_projections/'
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    else:
        shutil.rmtree(encoder_dir)
        os.makedirs(encoder_dir)

    # plot loss and accuracy as a function of time
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(len(val_loss))
    try:
        plot_loss(epochs, loss, val_loss, root + '_encoder_projections/autoencoder_')
    except:
        print('too many epochs')

    encoder = load_model(r'./weights_' + root + '/encoder_weights.h5')
    decoder = load_model(r'./weights_' + root + '/decoder_weights.h5')
    test_X = np.array(test_X)

    # auto-encoder predictions
    x = encoder.predict(test_X)
    y = decoder.predict(x)

    # accuracy
    calc_accuracy_zero_padding(test_X, y, encoder_dir)

    # Classifier ----------------------------------------------

    # train + test sets
    # train_X, test_X, train_Y, test_Y = train_test_split(vecs_data, tags, test_size=0.20)
    train_X, test_X, train_Y, test_Y, test_files = my_train_test_split(vecs_data, tags, files, test_size=0.20)

    # train classifier model
    ae.classifier()
    # use the auto-encoder layers in the classifier
    ae.encoder_2.set_weights(ae.encoder.get_weights())

    classify_train = ae.fit_classifier(np.array(train_X), np.array(train_Y), batch_size=50, epochs=500)  # 300
    ae.save_cl()

    # plot loss and accuracy as a function of time
    accuracy = classify_train.history['acc']
    val_accuracy = classify_train.history['val_acc']
    loss = classify_train.history['loss']
    val_loss = classify_train.history['val_loss']
    epochs = range(len(accuracy))
    try:
        plot_loss_acc(epochs, accuracy, val_accuracy, loss, val_loss, root + '_encoder_projections/classifier_')
    except:
        print('too many epochs')

    classifier = load_model(r'./weights_' + root + '/classifier_weights.h5')
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # classifier predictions
    preds = classifier.predict(test_X)
    preds = [i[0] for i in preds]

    calc_accuracy_classifier(test_Y, preds, root + '_encoder_projections/')
    calc_accuracy_by_k_classifier(test_Y, preds, [5, 10, 20], root + '_encoder_projections/')
    calc_accuracy_by_sample(test_Y, preds, test_files, 5, root + '_encoder_projections/')

    auc_, fpr_, tpr_ = calc_auc(test_Y, preds)

    plot_auc({'auc': auc_, 'fpr': fpr_, 'tpr': tpr_}, root, encoder_dir)

    # Embedding Auto-encoder ----------------------------------------------

    # train + test sets
    train_X, test_X, train_y, test_y = train_test_split(vecs_data, vecs_data, test_size=0.2)
    train_X = np.array(train_X)

    # calculate D- input distances matrix- norm 2
    D = spatial.distance.cdist(train_X, train_X, 'euclidean')

    # train model
    ae = Embedding_AutoEncoder(train_X, D, root, encoding_dim=50, batch_size=50, emb_alpha=0.01)
    ae.encoder_decoder()
    embedding_train = ae.fit_generator(epochs=100)  # 300
    ae.save()

    embedding_dir = root + '_embedding_projections/'
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    else:
        shutil.rmtree(embedding_dir)
        os.makedirs(embedding_dir)

    # # plot loss and accuracy as a function of time
    # loss = embedding_train.history['loss']
    # val_loss = embedding_train.history['val_loss']
    # epochs = range(len(val_loss))
    # try:
    #     plot_loss(epochs, loss, val_loss, embedding_dir)
    # except:
    #     print('too many epochs')

    # test model
    encoder = load_model(r'./weights_' + root + '/embedding_encoder_weights.h5')
    decoder = load_model(r'./weights_' + root + '/embedding_decoder_weights.h5')
    test_X = np.array(test_X)

    input_vec_size = len(test_X[0])
    cdr3_len = input_vec_size/21
    pickle.dump(cdr3_len, open('cdr3_len.p', "wb"))

    x = encoder.predict(test_X)
    y = decoder.predict(x)

    # accuracy
    calc_accuracy_zero_padding(test_X, y, embedding_dir)

    # Save Projections ----------------------------------------------

    # load CDR3 size for the autoencoder
    n = int(pickle.load(open("cdr3_len.p", "rb"))) - 1  # the stop sequence: !

    # predict auto-encoder representation
    samples = load_all_data(root, n, ',', 'Chain 2 CDR3 Curated', 'Calculated Chain 1 V Gene', 'Calculated Chain 1 J Gene')

    encoder = load_model(r'./weights_' + root + '/encoder_weights.h5')
    embedding_encoder = load_model(r'./weights_' + root + '/embedding_encoder_weights.h5')

    for s in samples:
        x_encoder = encoder.predict(np.array(samples[s]['vecs']))
        x_embedding = embedding_encoder.predict(np.array(samples[s]['vecs']))
        info = [[x_encoder[i], samples[s]['seqs'][i], samples[s]['vs'][i], samples[s]['js'][i]]
                for i in range(len(x_encoder))]
        pickle.dump(info, open(root + '_encoder_projections/' + str(s) + '.p', "wb"))
        info = [[x_embedding[i], samples[s]['seqs'][i], samples[s]['vs'][i], samples[s]['js'][i]]
                for i in range(len(x_embedding))]
        pickle.dump(info, open(root + '_embedding_projections/' + str(s) + '.p', "wb"))

    # TSNE ----------------------------------------------

    path = root + '_encoder_projections/'
    vecs_dict = load_representations(path)
    inds = []
    for i, f in enumerate(vecs_dict):
        if i == 0:
            X = vecs_dict[f]
        else:
            X = np.concatenate((X, vecs_dict[f]), axis=0)
        inds += [f] * len(vecs_dict[f])
    T = TSNE(n_components=2).fit_transform(X)
    scatter_tsne(T, inds, 'c', 0, 'Cancerous', 'Healthy', os.path.join(path, 'tsne_results'))

    # KDE ----------------------------------------------

    path = root + '_embedding_projections/'
    vecs_dict = load_representations(path)

    # compute distance between every two samples
    dis_dict = dict()
    for i, file in enumerate(vecs_dict):
        dis_dict[file] = dict()
        for j, file2 in enumerate(vecs_dict):
            # non-simetric distance metric
            if i == j:
                # keep seperate dict for self distances for bar plot
                dis_dict[file][file2] = kde_distance(vecs_dict[file], vecs_dict[file2], flag=True)
            else:
                dis_dict[file][file2] = kde_distance(vecs_dict[file], vecs_dict[file2])
            # if i < j:
            #     dis_dict[file][file2] = kde_distance(vecs_dict[file], vecs_dict[file2])
            # else:
            #     dis_dict[file][file2] = dis_dict[file2][file]

    headers = list(dis_dict.keys())

    # fill the full matrix of the data
    with open(path + 'kde_representations.csv', 'w') as csvfile:
        fieldnames = ['file'] + headers
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data_arr = []
        self_arr = []
        for i, f in enumerate(headers):
            data = {'file': f}
            tmp = []
            for j, f2 in enumerate(headers):
                val = dis_dict[f][f2]
                # keep the real value in the csv file
                data[f2] = val
                # remove only from the heatmap
                if i == j:
                    self_arr.append(val)
                    # val = np.nan
                tmp.append(val)
            writer.writerow(data)
            data_arr.append(tmp)

    data_arr = []
    self_arr = []
    with open(path + 'kde_representations.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            del row['file']
            headers = list(row.keys())
            tmp = []
            for j, key in enumerate(row):
                val = np.float(row[key])
                if i == j:
                    self_arr.append(val)
                    tmp.append(np.nan)
                else:
                    tmp.append(val)
            data_arr.append(tmp)

    # order = []
    # inds = []
    # for i, h in enumerate(headers):
    #     if 'CD4' in h:
    #         order.append(h)
    #         inds.append(i)
    # for i, h in enumerate(headers):
    #     if 'CD8' in h:
    #         order.append(h)
    #         inds.append(i)
    #
    # data_order = []
    # self_order = []
    # for i in inds:
    #     tmp = []
    #     self_order.append(self_arr[i])
    #     for j in inds:
    #         tmp.append(data_arr[i][j])
    #     data_order.append(tmp)

    plot_heatmap(data_arr, self_arr, headers, path)
    plot_self_bar(self_arr, headers, path)

    # Projections Properties ----------------------------------------------

    # Public Clones

    path = root + '_encoder_projections/'
    public_cdr3_dict = public_cdr3s(root, ',', 'Chain 2 CDR3 Curated')
    projections_dict = load_representations_and_seqs(path)
    projections_arr = []
    for s in projections_dict:
        projections_arr += projections_dict[s]
    center_p = find_center(projections_arr)
    public_distances = distances_from_center(projections_arr, center_p, public_cdr3_dict)
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

    path = root + '_encoder_projections/'
    projections_dict = load_representations_and_seqs(path)
    vecs = []
    seqs = []
    for s in projections_dict:
        vecs += [v[0] for v in projections_dict[s]]
        seqs += [v[1] for v in projections_dict[s]]

    T = TSNE(n_components=2).fit_transform(np.array(vecs))

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

    scatter_tsne_all_lengths(T, seqs, path)

    CM = np.average(vecs, axis=0)
    lens = []
    dis = []
    for i in range(len(vecs)):
        lens.append(len(seqs[i]))
        dis.append(np.linalg.norm(vecs[i] - CM))
    df = pd.DataFrame(data={'CDR3 Length': lens, 'Distance From Center': dis})
    scatter_cm(df, path)






