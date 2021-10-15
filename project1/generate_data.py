from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

N = 100000

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

df = pd.concat(features)
