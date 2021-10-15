from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from pipelinehelper import PipelineHelper
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.utils import resample


def generate_data(N=100000):
    _feature_covid_recovered = np.random.binomial(1, 0.03, N)[:, np.newaxis]
    _feature_covid_positive = np.random.binomial(1, 0.3, N)[:, np.newaxis]

    _feature_symptoms = np.random.randint(
        low=0,
        high=8,
        size=N
    )[:, np.newaxis]

    _feature_symptoms = \
        OneHotEncoder(sparse=False).fit_transform(_feature_symptoms)[:, 1:]

    _feature_death = np.zeros(N)[:, np.newaxis]
    _feature_ages = np.random.randint(1, 100, N)[:, np.newaxis]
    _feature_gender = np.random.binomial(1, 0.5, N)[:, np.newaxis]

    _feature_income = np.random.normal(25000, 10000, N)[:, np.newaxis]
    _feature_income[_feature_income <= 10] = 0

    _feature_genes = {}
    for i in range(128):
        _feature_genes[f'g{i+1}'] = np.random.binomial(1, 0.25, N)

    _feature_asthma = np.random.binomial(1, 0.07, N)[:, np.newaxis]
    _feature_obesity = np.random.binomial(1, 0.13, N)[:, np.newaxis]
    _feature_smoking = np.random.binomial(1, 0.19, N)[:, np.newaxis]
    _feature_diabetes = np.random.binomial(1, 0.10, N)[:, np.newaxis]
    _feature_heart_disease = np.random.binomial(1, 0.10, N)[:, np.newaxis]
    _feature_hypertension = np.random.binomial(1, 0.17, N)[:, np.newaxis]

    _feature_vaccines = np.random.randint(
        low=0,
        high=4,
        size=N
    )[:, np.newaxis]

    _feature_vaccines = \
        OneHotEncoder(sparse=False).fit_transform(_feature_vaccines)[:, 1:]

    features = [
        pd.DataFrame(_feature_covid_recovered, columns=['Covid-Recovered']),
        pd.DataFrame(_feature_covid_positive, columns=['Covid-Positive']),
        pd.DataFrame(_feature_symptoms, columns=['No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', ]),
        pd.DataFrame(_feature_death, columns=['Death']),
        pd.DataFrame(_feature_ages, columns=['Age']),
        pd.DataFrame(_feature_gender, columns=['Gender']),
        pd.DataFrame(_feature_income, columns=['Income']),
        pd.DataFrame(_feature_genes),
        pd.DataFrame(_feature_asthma, columns=['Asthma']),
        pd.DataFrame(_feature_obesity, columns=['Obesity']),
        pd.DataFrame(_feature_smoking, columns=['Smoking']),
        pd.DataFrame(_feature_diabetes, columns=['Diabetes']),
        pd.DataFrame(_feature_heart_disease, columns=['Heart-disease']),
        pd.DataFrame(_feature_hypertension, columns=['Hypertension']),
        pd.DataFrame(_feature_vaccines, columns=['Vaccine1', 'Vaccine2', 'Vaccine3']),
    ]

    df = pd.concat(features, axis=1)

    # fixing income to zero for people below 18 years old
    df['Income'].where(df['Age'] > 18, 0, inplace=True)
    return df


def defining_conditional_probabilities():
    ages_ = np.array([20, 40, 60, 80, 100])
    diabetes_ = np.array([0, 1])
    hypertension_ = np.array([0, 1])
    g1g2_ = np.array([0, 2])
    v1_ = np.array([0, 1])
    v2_ = np.array([0, 1])
    v3_ = np.array([0, 1])

    cond_probs = np.array(
        np.meshgrid(ages_, diabetes_, hypertension_, g1g2_, v1_, v2_, v3_)
    ).T.reshape(-1, 7)

    prob_ages_ = {20: 0.05, 40: 0.15, 60: 0.25, 80: 0.50, 100: 0.70}
    prob_diabetes_ = {0: 0., 1: 0.30}
    prob_hypertension_ = {0: 0., 1: 0.30}
    prob_g1g2_ = {0: 0., 2: 0.90}
    prob_v1_ = {0: 0., 1: -0.30}
    prob_v2_ = {0: 0., 1: -0.40}
    prob_v3_ = {0: 0., 1: -0.50}

    probs = np.zeros((len(cond_probs),))
    idx_probs = 0
    for idx_r, r in enumerate(cond_probs):

        p = prob_ages_[cond_probs[idx_r, 0]] + \
            prob_diabetes_[cond_probs[idx_r, 1]] + \
            prob_hypertension_[cond_probs[idx_r, 2]] + \
            prob_g1g2_[cond_probs[idx_r, 3]] + \
            prob_v1_[cond_probs[idx_r, 4]] + \
            prob_v2_[cond_probs[idx_r, 5]] + \
            prob_v3_[cond_probs[idx_r, 6]]

        if p <= 0.01:
            probs[idx_probs] = 0.01
        elif p >= 0.9:
            probs[idx_probs] = 0.9
        else:
            probs[idx_probs] = p
        idx_probs += 1

    cond_probs = pd.concat(
        [
            pd.DataFrame(
                cond_probs,
                columns=['Age', 'Diabetes', 'Hypertension',
                         'G1+G2', 'V1', 'V2', 'V3']
            ),
            pd.DataFrame(
                probs[:, np.newaxis],
                columns=['Probabilities']
            )
        ], axis=1)

    cond_probs_dict = {}
    for i in range(len(cond_probs)):
        key = f"{int(cond_probs.iloc[i, :]['Age'])}" + \
              f"{int(cond_probs.iloc[i, :]['Diabetes'])}" + \
              f"{int(cond_probs.iloc[i, :]['Hypertension'])}" + \
              f"{int(cond_probs.iloc[i, :]['G1+G2'])}" + \
              f"{int(cond_probs.iloc[i, :]['V1'])}" + \
              f"{int(cond_probs.iloc[i, :]['V2'])}" + \
              f"{int(cond_probs.iloc[i, :]['V3'])}"

        cond_probs_dict[key] = cond_probs.iloc[i, -1]

    return cond_probs, cond_probs_dict


def help_age(given_age):
    if given_age <= 20:
        return 20
    elif given_age <= 40:
        return 40
    elif given_age <= 60:
        return 60
    elif given_age <= 80:
        return 80
    else:
        return 100


def help_gene(given_gene):
    if given_gene == 1:
        return 0
    else:
        return given_gene


def assigning_death(df, cond_probs_dict):
    new_death = []
    for i in range(len(df)):
        if df.iloc[i]['Covid-Positive'] == 1:
            find = f"{int(help_age(given_age=df.iloc[i]['Age']))}" + \
                   f"{int(df.iloc[i]['Diabetes'])}" + \
                   f"{int(df.iloc[i]['Hypertension'])}" + \
                   f"{int(help_gene(given_gene=df.iloc[i]['g1'] + df.iloc[i]['g2']))}" + \
                   f"{int(df.iloc[i]['Vaccine1'])}" + \
                   f"{int(df.iloc[i]['Vaccine2'])}" + \
                   f"{int(df.iloc[i]['Vaccine3'])}"
            new_death.append(np.random.binomial(1, cond_probs_dict[find]))

        else:
            new_death.append(np.random.binomial(1, 0.05))

    df['Death'] = new_death
    return df


if __name__ == '__main__':
    print("Generating Data....\n")
    data = generate_data(N=100000)
    cond_probs, cond_probs_dict = defining_conditional_probabilities()
    df = assigning_death(df=data, cond_probs_dict=cond_probs_dict)
    #corr_mtx = df.corr()

    print("Fitting Model....\n")
    #pca = PCA()
    #pca.fit(df)

    #print(pd.DataFrame(pca.components_,columns=df.columns))

    df_dead = df[df["Death"] == 1.0]
    df_not_dead = df[df["Death"] == 0.0].iloc[:df_dead.shape[0], :]
    df_balanced = pd.concat([df_dead, df_not_dead])

    clf = LogisticRegressionCV(penalty='l2',max_iter=200)

    clf.fit(df_balanced.drop(["Death"], axis=1).to_numpy(), df_balanced["Death"])
    idx = 0
    for c in clf.coef_[0]:
        print(idx, c)
        idx += 1
    print(clf.scores_)