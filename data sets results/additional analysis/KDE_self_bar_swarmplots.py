import os
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)


def csv_to_list(path_to_csv_):
    headers_ = []
    self_arr_ = []
    with open(path_to_csv_, mode='r') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            del row['file']
            if i == 0:
                headers_ = list(row.keys())
            for j, key in enumerate(row):
                val = np.float(row[key])
                if i == j:
                    self_arr_.append(np.log(val))
    return self_arr_, headers_


def plot_categories_swarmplot(df, path, encoder, round):
    sns.set(font_scale=1.3)
    s = sns.swarmplot(x='Category', y='Self-density', data=df, order=['Naive', 'Memory', 'Alpha', 'Beta', 'CD4', 'CD8'])
    plt.xticks(fontsize=13)
    plt.title('Self-density By Category, ' + encoder + ' ' + round, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'kde_self_bar_swarmplot_' + encoder + '_' + round +'.png'))
    plt.close()


if __name__ == '__main__':

    root = 'benny_chain_processed_data'
    save_path = root + '_Results'

    all_encoders = [os.path.join(save_path, s + '_projections/') for s in
                    ['encoder', 'v_encoder', 'embedding', 'v_embedding']]

    methods = {}
    for path in all_encoders:

        for round in ['no_freqs', 'with_freqs']:

            path_to_csv = path + 'kde_representations_' + round + '.csv'
            my_file = Path(path_to_csv)

            if my_file.exists():

                diagonal, files = csv_to_list(path_to_csv)

                groups = {'Naive': [], 'Memory': [], 'CD4': [], 'CD8': [], 'Alpha': [], 'Beta': []}
                for i, f in enumerate(files):
                    val = diagonal[i]
                    if 'naive' in f:
                        groups['Naive'].append(val)
                    else:
                        groups['Memory'].append(val)
                    if 'CD4' in f:
                        groups['CD4'].append(val)
                    else:
                        groups['CD8'].append(val)
                    if 'alpha' in f:
                        groups['Alpha'].append(val)
                    else:
                        groups['Beta'].append(val)
                x = []
                y = []
                for key in groups:
                    for val in groups[key]:
                        x.append(key)
                        y.append(val)
                d = {'Category': x, 'Self-density': y}
                df = pd.DataFrame(data=d)
                model = path.split(os.sep)[1]
                model = '_'.join(model.split('_')[:-1])
                plot_categories_swarmplot(df, save_path, model, round)

                methods[model + '_' + round] = groups

    x = []
    y = []
    z = []
    for method in methods:
        for category in methods[method]:
            for val in methods[method][category]:
                x.append(method)
                y.append(category)
                z.append(val)
    d = {'Method': x, 'Category': y, 'value': z}
    df = pd.DataFrame(data=d)
    model = ols('value ~ C(Method) + C(Category) + C(Method):C(Category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    MultiComp = MultiComparison(df['value'], df['Method'])
    print(MultiComp.tukeyhsd().summary())

    MultiComp = MultiComparison(df['value'], df['Category'])
    print(MultiComp.tukeyhsd().summary())



