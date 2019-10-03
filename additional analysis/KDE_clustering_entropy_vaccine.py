import os
from matplotlib import pyplot as plt
import csv
import seaborn as sns
import numpy as np
from pathlib import Path
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sc
from scipy.stats import entropy
import pandas as pd
from sklearn.manifold import MDS


def my_entropy(labels, base=None):
  value, counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)


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


def plot_heatmap(data, diagonal, headers, path):
    sns.set(font_scale=0.5)
    for i, row in enumerate(data):
        for j in range(len(row)):
            if i != j:
                data[i][j] = data[i][j]/diagonal[i]
    sns.heatmap(data, yticklabels=headers, xticklabels=['']*len(headers), cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot_kws={"size": 10})
    plt.title('KDE Representations Distances')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
    return data


def opposite_distances(data):
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                data[i][j] = 1/data[i][j]
    return data


def rename_headers(files):
    names = {}
    with open('vaccine_TCR_names.csv', mode='r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for row in reader:
            names[row['SRA']] = row['NAME']

    for i, file in enumerate(files):
        file = file.split('.')[0]
        sra = file.split('_')[0]
        host = names[sra].split('_')[0]
        files[i] = file.replace(sra, host)

    return files


def plot_cluster_gram(my_data, my_headers, path):
    for i in range(len(my_data)):
        for j in range(len(my_data)):
            if i == j:
                my_data[i][i] = 0
            else:
                if i < j:
                    my_data[i][j] = my_data[j][i]
    distArray = ssd.squareform(my_data)
    linkage = sc.linkage(distArray, method='average')
    d = sc.dendrogram(linkage, labels=my_headers, above_threshold_color="grey",
                      orientation='right', leaf_font_size=7)
    clusters = {i: [file] for i, file in enumerate(my_headers)}
    ind = len(clusters)
    for i in range(len(linkage)):
        if len(clusters) == 6:
            break
        c1 = int(linkage[i, 0])
        c2 = int(linkage[i, 1])
        clusters[i + ind] = clusters[c1] + clusters[c2]
        del clusters[c1]
        del clusters[c2]

    order = [my_headers[i] for i in d['leaves']]
    plt.title('KDE Dendogram')
    plt.savefig(path)
    plt.close()
    return order, clusters


def plot_categories_swarmplot(df, type):
    sns.set(font_scale=0.8)
    sns.swarmplot(x='Run', y='Clusters Entropies', data=df)
    plt.xticks(rotation=90)
    # g.set_xticklabels(labels)
    plt.title('Clusters Entropies By ' + type)
    plt.tight_layout()
    plt.savefig('entropies_swarmplot_' + type + '.png')
    plt.close()


def plot_radius_swarmplot(df, path, type, order):
    sns.set(font_scale=0.8)
    if len(order) == 0:
        sns.swarmplot(x='Run', y='MDS Radius', data=df)
    else:
        sns.swarmplot(x='Run', y='MDS Radius', data=df, order=order)
    # plt.xticks(rotation=90)
    # g.set_xticklabels(labels)
    plt.title('MDS Radius By ' + type)
    plt.tight_layout()
    plt.savefig(path + '_' + type + '.png')
    plt.close()


def clustering(path_to_csv, path_to_heatmap, path_to_dendogram, reverse=False):

    data_arr, self_arr, headers = csv_to_arr(path_to_csv)
    headers = rename_headers(headers)
    if reverse:
        data_arr = plot_heatmap(data_arr, self_arr, headers, path_to_heatmap)
        data_arr = opposite_distances(data_arr)

    leaves_order, cs = plot_cluster_gram(data_arr, headers, path_to_dendogram)
    hosts = {}
    for c in cs:
        hosts[c] = [i.split('_')[0] for i in cs[c]]
    vals_hosts = []
    for cluster in hosts:
        vals_hosts.append(my_entropy(hosts[cluster]))

    leaves_order, cs = plot_cluster_gram(data_arr, headers, path_to_dendogram)
    tps = {}
    for c in cs:
        tps[c] = ['_'.join(i.split('_')[1:]) for i in cs[c]]
    vals_tps = []
    for cluster in tps:
        vals_tps.append(my_entropy(tps[cluster]))

    return vals_hosts, vals_tps


def bar_plot(y_list, x_ticks, path, type):
    pos = np.arange(len(x_ticks))
    plt.bar(pos, y_list, align='center', alpha=0.5, color='red')
    plt.xticks(pos, x_ticks, fontsize=10)
    plt.ylabel('Average Radius')
    plt.title('MDS Radius')
    plt.tight_layout()
    plt.savefig(path + '_' + type + '.png')
    plt.clf()
    plt.close()


def MDS_radius(path_to_csv, path_to_bar, path_to_swarmplot, reverse=False):

    data_arr, self_arr, headers = csv_to_arr(path_to_csv)
    headers = rename_headers(headers)
    if reverse:
        data_arr = plot_heatmap(data_arr, self_arr, headers, path_to_heatmap)
        data_arr = opposite_distances(data_arr)

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

    model = MDS(n_components=2, random_state=1)
    M = model.fit_transform(data_arr)

    CM = np.average(M, axis=0)
    hosts_rs = {}
    tps_rs = {}
    x_host = []
    x_tp = []
    y = []
    for i, h in enumerate(headers):
        dis = np.linalg.norm(M[i] - CM)
        host, tp = h.split('_', 1)
        hosts_rs[host] = hosts_rs.get(host, []) + [dis]
        tps_rs[tp] = tps_rs.get(tp, []) + [dis]
        x_host.append(host)
        x_tp.append(tp)
        y.append(dis)

    d = {'Run': x_host, 'MDS Radius': y}
    df = pd.DataFrame(data=d)
    plot_radius_swarmplot(df, path_to_swarmplot, 'hosts', [])

    order = ['7_D_Pre', '0_D_Post', '7_D_Post', '15_D_Post', '45_D_Post', '2_Y_Post']

    d = {'Run': x_tp, 'MDS Radius': y}
    df = pd.DataFrame(data=d)
    plot_radius_swarmplot(df, path_to_swarmplot, 'tps', order)

    hosts_rs = {key: np.average(val) for key, val in hosts_rs.items()}
    tps_rs = {key: np.average(val) for key, val in tps_rs.items()}

    bar_plot(hosts_rs.values(), hosts_rs.keys(), path_to_bar, 'hosts')
    bar_plot([tps_rs[key] for key in order], order, path_to_bar, 'tps')

    return hosts_rs, tps_rs


if __name__ == '__main__':

    root = 'vaccine_TCR_processed_data'
    save_path = root + '_Results'

    encoder_dir = os.path.join(save_path, 'encoder_projections/')
    v_encoder_dir = os.path.join(save_path, 'v_encoder_projections/')
    embedding_dir = os.path.join(save_path, 'embedding_projections/')
    v_embedding_dir = os.path.join(save_path, 'v_embedding_projections/')

    clusters_hosts = {}
    clusters_tps = {}
    all_rs_host = {}
    all_rs_tps = {}

    # Encoders

    for path in [encoder_dir, v_encoder_dir, embedding_dir]:

        for round in ['no_freqs', 'with_freqs']:

            path_to_csv = path + 'kde_representations_' + round + '.csv'
            my_file = Path(path_to_csv)

            if my_file.exists():

                path_to_heatmap = path + 'kde_representations_distances_' + round + '.png'
                path_to_dendogram = path + 'kde_dendogram_' + round + '.png'
                path_to_bar = path + 'mds_radius_' + round
                path_to_swarmplot = path + 'swarmplot_mds_radius_' + round
                key = os.path.normpath(path).split(os.sep)[1] + '_' + round

                vals_hosts, vals_tps = clustering(path_to_csv, path_to_heatmap, path_to_dendogram, reverse=True)
                clusters_hosts[key] = vals_hosts
                clusters_tps[key] = vals_tps

                rs_hosts, rs_tps = MDS_radius(path_to_csv, path_to_bar, path_to_swarmplot, reverse=True)
                all_rs_host[key] = rs_hosts
                all_rs_tps[key] = rs_tps

    # KL

    path_to_csv = os.path.join(save_path, 'kl_distances.csv')
    my_file = Path(path_to_csv)

    if my_file.exists():
        path_to_heatmap = 'kl_representations_distances.png'
        path_to_dendogram = 'kl_dendogram.png'
        path_to_bar = os.path.join(save_path, 'kl_mds_radius')
        path_to_swarmplot = os.path.join(save_path, 'swarmplot_kl_mds_radius')
        key = 'kl'

        vals_hosts, vals_tps = clustering(path_to_csv, path_to_heatmap, path_to_dendogram)
        clusters_hosts[key] = vals_hosts
        clusters_tps[key] = vals_tps

        rs_hosts, rs_tps = MDS_radius(path_to_csv, path_to_bar, path_to_swarmplot)
        all_rs_host[key] = rs_hosts
        all_rs_tps[key] = rs_tps

    # ED

    path_to_csv = os.path.join(save_path, 'ed_distances.csv')
    my_file = Path(path_to_csv)

    if my_file.exists():
        path_to_heatmap = 'ed_representations_distances.png'
        path_to_dendogram = 'ed_dendogram.png'
        path_to_bar = os.path.join(save_path, 'ed_mds_radius')
        path_to_swarmplot = os.path.join(save_path, 'swarmplot_ed_mds_radius')
        key = 'ed'

        vals_hosts, vals_tps = clustering(path_to_csv, path_to_heatmap, path_to_dendogram)
        clusters_hosts[key] = vals_hosts
        clusters_tps[key] = vals_tps

        rs_hosts, rs_tps = MDS_radius(path_to_csv, path_to_bar, path_to_swarmplot)
        all_rs_host[key] = rs_hosts
        all_rs_tps[key] = rs_tps

    # Swarmplots

    x = []
    y = []
    for key in clusters_hosts:
        vals = clusters_hosts[key]
        x += [key]*len(vals)
        y += vals
    d = {'Run': x, 'Clusters Entropies': y}
    df = pd.DataFrame(data=d)
    plot_categories_swarmplot(df, 'hosts')

    x = []
    y = []
    for key in clusters_tps:
        vals = clusters_tps[key]
        x += [key] * len(vals)
        y += vals
    d = {'Run': x, 'Clusters Entropies': y}
    df = pd.DataFrame(data=d)
    plot_categories_swarmplot(df, 'time points')



