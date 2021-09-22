from pandas import read_csv
from seaborn import heatmap
from matplotlib.pyplot import show


def import_data():
    observation_features = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/observation_features.csv",
        header=None,
    )

    treatment_features = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_features.csv",
        header=None,
    )

    treatment_action = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_actions.csv",
        header=None,
    )

    treatment_outcome = read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/treatment_outcomes.csv",
        header=None,
    )

    return (observation_features, treatment_features, treatment_action,
            treatment_outcome)


def plot_heatmap_corr(df, labels, _show=False):
    plot = heatmap(
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

    return plot
