import matplotlib.pyplot as plt
from pandas import read_csv
from seaborn import heatmap
import seaborn as sns
import numpy as np
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
    treatment_action.columns = ['Treatment1', 'Treatment2']
    treatment_outcome.columns = labels[0:10]

    return (observation_features, treatment_features, treatment_action,
            treatment_outcome)


def plot_heatmap_corr(df, labels, _show=False, annot=False):
    """Show a correlation matrix."""
    heatmap(
        df.corr(),
        annot=annot,
        # yticklabels=labels,
        linewidths=.5,
    ).set_xticklabels(
        labels,
        rotation=90,
    )
    # sns.set(font_scale=0.7)
    plt.show() if _show is True else None


def age_analysis(df, plot_box=False, plot_dist=False):
    """Return a age analysis."""
    # How many of death per age?
    df[10].plot.box()
    plt.show() if plot_box is True else None

    # people with > 120 years old?
    print(f"Number of deaths with over 120 years old: "
          f"{df[10][df[10] > 120].count()}")

    # age mean of the deaths
    print(f"Age mean of the deaths: "
          f"{df[10][df[9] == 1.0].mean()}")

    # density of the ages given they are dead
    df[10][df[9] == 1.0].plot.density()
    plt.show() if plot_dist is True else None
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


def feature_importance_methodology1(syn_neg_corr, syn_pos_corr):
    print('Plotting...')
    sns.barplot(
        y=pd.concat([syn_neg_corr, syn_pos_corr], axis=0)[:-1].values,
        x=pd.concat([syn_neg_corr, syn_pos_corr], axis=0)[:-1].index,
    ).figure.subplots_adjust(left=0.15, bottom=0.3)
    plt.xticks(rotation=90)
    plt.title("Top 15 negative and positive correlations with 'Death'")
    plt.ylabel('Correlation scores')
    plt.show()
    print('done.')


def feature_importance_methodology3(best_model, topn=15):
    arg_sorted = np.argsort(best_model.coef_.ravel())

    top_pos_vals = best_model.coef_.ravel()[arg_sorted][-topn:]
    top_pos_names = best_model.feature_names_in_[arg_sorted][-topn:]

    top_neg_vals = best_model.coef_.ravel()[arg_sorted][:topn]
    top_neg_names = best_model.feature_names_in_[arg_sorted][:topn]

    print('Plotting...')
    sns.barplot(
        x=top_neg_vals,
        y=top_neg_names,
    ).figure.subplots_adjust(left=0.2, bottom=0.2)
    plt.xticks(rotation=90)
    plt.title(
        "Top 15 negative coefficients from the best Logistic Regression model")
    plt.xlabel('Coefficients')
    plt.show()
    print('done.')

    print('Plotting...')
    sns.barplot(
        x=top_pos_vals,
        y=top_pos_names,
    ).figure.subplots_adjust(left=0.2, bottom=0.2)
    plt.xticks(rotation=90)
    plt.title(
        "Top 15 positive coefficients from the best Logistic Regression model")
    plt.xlabel('Coefficients')
    plt.show()
    print('done.')
