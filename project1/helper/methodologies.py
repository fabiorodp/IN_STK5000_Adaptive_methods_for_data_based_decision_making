try:
    from generate_data import Space
except:
    from project1.helper.generate_data import Space

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def methodology1(data: pd.DataFrame, parameter='Death'):
    """Function to calculate the correlations among features."""
    df = data.corr()

    neg_corr = df[parameter].sort_values().head(15)
    print(df[parameter].sort_values().head(15))

    pos_corr = df[parameter].sort_values().tail(15)
    print(df[parameter].sort_values().tail(15))
    return neg_corr, pos_corr


def methodology2(data: pd.DataFrame, explanatories, responses: list):
    """Function to calculate the conditional probability among features."""
    cond_dict_f_f = {}
    cond_dict_f_t = {}
    cond_dict_t_f = {}
    cond_dict_t_t = {}
    for r in responses:
        for e in explanatories:
            cond_dict_f_f[f'{r}|{e}'] = []
            cond_dict_f_t[f'{r}|{e}'] = []
            cond_dict_t_f[f'{r}|{e}'] = []
            cond_dict_t_t[f'{r}|{e}'] = []

    for i in range(10):
        sample_data = data.sample(frac=0.25, replace=True)
        for r in responses:
            for e in explanatories:
                _prob_explanatory = sample_data.groupby([e]).size().div(
                    len(sample_data))  # P(explanatory)
                _prob_response_explanatory = sample_data.groupby(
                    [r, e]).size().div(
                    len(sample_data))  # P(response, explanatory)
                _cprob = _prob_response_explanatory.div(_prob_explanatory,
                                                        axis=0)

                cond_dict_f_f[f'{r}|{e}'].append(_cprob.iloc[0])
                print(f'P(not {r}| not {e}) = {_cprob.iloc[0]}')

                cond_dict_f_t[f'{r}|{e}'].append(_cprob.iloc[1])
                print(f'P(not {r}| {e})  {_cprob.iloc[1]}')

                try:
                    cond_dict_t_f[f'{r}|{e}'].append(_cprob.iloc[2])
                    print(f'P({r}| not {e}) = {_cprob.iloc[2]}')
                except:
                    cond_dict_t_f[f'{r}|{e}'].append(0)
                    print(f'P({r}| not {e}) = {0}')

                try:
                    cond_dict_t_t[f'{r}|{e}'].append(_cprob.iloc[3])
                    print(f'P({r}|{e}) = {_cprob.iloc[3]}')
                except:
                    cond_dict_t_t[f'{r}|{e}'].append(0)
                    print(f'P({r}|{e}) = {0}')

    conds = [cond_dict_f_f, cond_dict_f_t, cond_dict_t_f, cond_dict_t_t]
    labels = ['False|False', 'False|True', 'True|False', 'True, True']

    for a, b in zip(conds, labels):
        sns.boxplot(
            data=pd.DataFrame(a),
            orient='h'
        ).figure.subplots_adjust(left=0.35, bottom=0.15)
        plt.title(f'Conditional Probabilities {b}')
        plt.xlabel("P")
        plt.show()


def methodology3(X: pd.DataFrame, Y: pd.DataFrame,
                 max_iter: int, cv: int, seed=1, n_jobs=-1,
                 model_type='LR', score_type='accuracy'):
    """Function creating a pipeline with a Logistic Regression model
    to estimate a parameter."""
    pipeline, param_dist = '', ''
    if model_type == 'LR':
        steps = [
            ('lr', LogisticRegression(
                solver='saga',
                random_state=seed,
                n_jobs=n_jobs,
                verbose=False
            ))
        ]

        pipeline = Pipeline(steps)

        param_dist = {
            'lr__penalty': ['l1', 'l2'],
            'lr__max_iter': [5, 10, 15, 20, 25, 50],
            'lr__C': [0.1, 0.25, 0.5, 0.75, 1],
        }

    elif model_type == 'RF':
        steps = [
            ('rf', RandomForestClassifier(
                random_state=seed,
                n_jobs=n_jobs,
                verbose=False
            ))
        ]

        pipeline = Pipeline(steps)

        param_dist = {
            'rf__n_estimators': [50, 100, 500, 1000],
            'rf__criterion': ['gini', 'entropy'],
            'rf__bootstrap': [True, False],
        }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=max_iter,
        scoring=score_type,
        refit=True,
        cv=cv,
        random_state=seed
    )
    search.fit(X, Y)
    print(f'Best Cross-Validated mean score: {search.best_score_}')
    return search
