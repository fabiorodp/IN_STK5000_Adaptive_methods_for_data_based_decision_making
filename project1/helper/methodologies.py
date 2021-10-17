try:
    from generate_data import Space
except:
    from project1.helper.generate_data import Space
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def methodology1(data: pd.DataFrame):
    """Function to calculate the correlations among features."""
    df = data.corr()
    neg_corr = df['Death'].sort_values().head(20)
    pos_corr = df['Death'].sort_values().tail(20)
    return neg_corr, pos_corr


def methodology2(data: pd.DataFrame, explanatories: list, responses: list):
    """Function to calculate the conditional probability among features."""
    for e in responses:
        for v in explanatories:
            _prob = data.groupby([e]).size().div(len(data))
            conditional_prob = data.groupby([e, v]).size().div(len(data)).div(
                _prob, axis=0, level=e)
            print(f"P({e} | {v}) = {conditional_prob} \n")


def methodology3(X, Y, max_iter, cv, seed=1, n_jobs=-1):
    """Function creating a pipeline with a Logistic Regression model
    to estimate a parameter."""

    # create pipeline with a PCA
    steps = [
        # ('pca', PCA()),
        ('lr', LogisticRegression(
            solver='saga',
            random_state=seed,
            n_jobs=n_jobs
        ))
    ]

    pipeline = Pipeline(steps)

    param_dist = {
        # 'PCA__n_components': [6, 10, 50, 100],
        'lr__penalty': ['l2'],
        'lr__max_iter': [10, 50, 200, 500, 1000],
        'lr__C': [0.1, 0.5, 1],
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
