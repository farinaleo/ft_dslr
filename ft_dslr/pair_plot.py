import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(dtf: pd.DataFrame):
    dtf = dtf.drop(columns='Index', inplace=False)
    main_column = 'Hogwarts House'

    _gryffindor = dtf[dtf[main_column] == 'Gryffindor'].copy()
    _hufflepuff = dtf[dtf[main_column] == 'Hufflepuff'].copy()
    _slytherin = dtf[dtf[main_column] == 'Slytherin'].copy()
    _ravenclaw = dtf[dtf[main_column] == 'Ravenclaw'].copy()

    sns.pairplot(dtf, hue=main_column, diag_kind='hist', plot_kws={"s": 3})
    plt.show()
