import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_column = 'Hogwarts House'


def pair_plot(dtf: pd.DataFrame):
    dtf = dtf.drop(columns='Index', inplace=False)
    nb_features = dtf.select_dtypes(include=['number']).columns.tolist()
    nb_features.reverse()
    feature_len = len(nb_features)
    color_map = {'Gryffindor': 'red', 'Ravenclaw': 'blue', 'Slytherin': 'green', 'Hufflepuff': 'yellow'}

    font = {'weight': 'light',
            'size': 7}
    plt.rc('font', **font)

    figure, axes = plt.subplots(nrows=feature_len, ncols=feature_len, squeeze=False)

    for x in range(0, feature_len):
        for y in range(0, feature_len):
            if nb_features[x] == nb_features[y]:
                plot_hist(dtf[[main_column, nb_features[x]]], axes[x, y], nb_features[x], nb_features[y])
            else:
                # axes[x, y].set_yticks(np.arange(0, 100, 4))
                # axes[x, y].set_xticks(np.arange(0, 100, 4))
                plot_scatter(dtf[[main_column, nb_features[x], nb_features[y]]], axes[x, y], nb_features[x], nb_features[y])
            if x != feature_len - 1:
                axes[x, y].tick_params(labelbottom=False)
                axes[x, y].xaxis.set_ticks_position('none')
            else:
                axes[x, y].set_xlabel(nb_features[y].replace(' ', '\n'))
                print(f' Feature y {nb_features[y]}')
                print(f'min: {dtf[nb_features[y]].min()} max: {dtf[nb_features[y]].max()}')
            if y != 0:
                axes[x, y].tick_params(labelleft=False)
                axes[x, y].yaxis.set_ticks_position('none')
            else:
                axes[x, y].set_ylabel(nb_features[x].replace(' ', '\n'), )
                print(f' Feature x {nb_features[x]}')
                print(f'min: {dtf[nb_features[x]].min()} max: {dtf[nb_features[x]].max()}')
            axes[x, y].tick_params(labelrotation=45.0)
            # axes[x, y].tick_params(labeltop=False)
            axes[x, y].tick_params(labelright=False)
            axes[x, y].tick_params(labeltop=False)
            axes[x, y].spines['right'].set_visible(False)
            axes[x, y].spines['top'].set_visible(False)


    plt.show()


def plot_scatter(dtf: pd.DataFrame, axe, x_name: str, y_name: str):
    _gryffindor = dtf[dtf[main_column] == 'Gryffindor'].copy()
    _hufflepuff = dtf[dtf[main_column] == 'Hufflepuff'].copy()
    _slytherin = dtf[dtf[main_column] == 'Slytherin'].copy()
    _ravenclaw = dtf[dtf[main_column] == 'Ravenclaw'].copy()

    axe.scatter(_gryffindor[x_name], _gryffindor[y_name], s=2, color='red', alpha=0.5)
    axe.scatter(_hufflepuff[x_name], _hufflepuff[y_name], s=2, color='yellow', alpha=0.5)
    axe.scatter(_slytherin[x_name], _slytherin[y_name], s=2, color='green', alpha=0.5)
    axe.scatter(_ravenclaw[x_name], _ravenclaw[y_name], s=2, color='blue', alpha=0.5)




def plot_hist(dtf: pd.DataFrame, axe, x_name: str, y_name: str):
    _gryffindor = dtf[dtf[main_column] == 'Gryffindor'][x_name]
    _hufflepuff = dtf[dtf[main_column] == 'Hufflepuff'][x_name]
    _slytherin = dtf[dtf[main_column] == 'Slytherin'][x_name]
    _ravenclaw = dtf[dtf[main_column] == 'Ravenclaw'][x_name]

    axe.hist(_gryffindor, color='red', label='1', alpha=0.5)
    axe.hist(_hufflepuff, color='yellow', label='2', alpha=0.5)
    axe.hist(_slytherin, color='green', label='3', alpha=0.5)
    axe.hist(_ravenclaw, color='blue', label='4', alpha=0.5)
    # axe.set_yticks(np.arange(_gryffindor[y_name].min(), _gryffindor[y_name].max(), 4))
    # axe.set_xticks(np.arange(_gryffindor[x_name].min(), _gryffindor[x_name].max(), 4))
