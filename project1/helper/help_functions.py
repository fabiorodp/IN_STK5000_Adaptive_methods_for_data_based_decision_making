from pandas import read_csv
from seaborn import heatmap
from matplotlib.pyplot import show
import pandas as pd


def import_data():
    """Import real data from GitHub."""
    observation_features = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/observation_features.csv.gz",
        header=None,
    )

    treatment_features = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_features.csv.gz",
        header=None,
    )

    treatment_action = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_actions.csv.gz",
        header=None,
    )

    treatment_outcome = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_outcomes.csv.gz",
        header=None,
    )

    labels = ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell',
              'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis',
              'Blood-Clots', 'Death', 'Age', 'Gender', 'Income'] + \
             [f'g{i}' for i in range(1, 129)] + \
             ['Asthma', 'Obesity', 'Smoking', 'Diabetes', 'Heart-disease',
              'Hypertension', 'Vaccine1', 'Vaccine2', 'Vaccine3']

    observation_features.columns = labels
    treatment_features.columns = labels
    treatment_action.columns = ['Treatment 1', 'Treatment 2']
    treatment_outcome.columns = labels[0:10]

    return (observation_features, treatment_features, treatment_action,
            treatment_outcome)


def plot_heatmap_corr(df, labels, _show=False):
    heatmap(
        df.corr(),
        annot=True,
        yticklabels=labels,
        linewidths=.5,
    ).set_xticklabels(
        labels,
        rotation=90,
    )
    # sns.set(font_scale=0.7)
    show() if _show is True else None


def age_analysis(df, plot_box=False, plot_dist=False):
    # How many of death per age?
    df[10].plot.box()
    show() if plot_box is True else None

    # people with > 120 years old?
    print(f"Number of deaths with over 120 years old: "
          f"{df[10][df[10] > 120].count()}")

    # age mean of the deaths
    print(f"Age mean of the deaths: "
          f"{df[10][df[9] == 1.0].mean()}")

    # density of the ages given they are dead
    df[10][df[9] == 1.0].plot.density()
    show() if plot_dist is True else None
    # kind of normal distribution


def balance_data(data, param='Death'):
    """Balancing targets."""
    df_dead = data[data[param] == 1.0]
    df_not_dead = data[data[param] == 0.0].iloc[
                  :df_dead.shape[0], :]
    df_balanced = pd.concat([df_dead, df_not_dead])
    df_balanced = df_balanced[df_balanced['Covid-Positive'] == 1].drop(
        ['Covid-Positive'], axis=1)
    return df_balanced
