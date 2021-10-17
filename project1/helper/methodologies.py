try:
    from generate_data import Space
except:
    from project1.helper.generate_data import Space
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def methodology1(data: pd.DataFrame, parameter='Death'):
    """Function to calculate the correlations among features."""
    df = data.corr()
    neg_corr = df[parameter].sort_values().head(20)
    pos_corr = df[parameter].sort_values().tail(20)
    return neg_corr, pos_corr


def methodology2(data: pd.DataFrame, explanatories, responses: list):
    """Function to calculate the conditional probability among features."""
    for r in responses:
        for v in explanatories:
            _prob = data.groupby([r]).size().div(len(data))
            conditional_prob = data.groupby([r, v]).size().div(len(data)).div(
                _prob, axis=0, level=r)
            print(f"P({r} | {v}) = {conditional_prob} \n")


def methodology3(X: pd.DataFrame, Y: pd.DataFrame,
                 max_iter: int, cv: int, seed=1, n_jobs=-1,
                 model_type='LR'):
    """Function creating a pipeline with a Logistic Regression model
    to estimate a parameter."""
    pipeline, param_dist = '', ''
    if model_type == 'LR':
        steps = [
            ('lr', LogisticRegression(
                solver='saga',
                random_state=seed,
                n_jobs=n_jobs
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
                n_jobs=n_jobs
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
        scoring='accuracy',
        refit=True,
        cv=cv,
        random_state=seed
    )
    search.fit(X, Y)
    return search
