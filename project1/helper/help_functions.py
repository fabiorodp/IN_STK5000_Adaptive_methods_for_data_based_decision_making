from pandas import read_csv
from seaborn import heatmap
from matplotlib.pyplot import show


def import_data():
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
