try:
    from api.users import credentials
    from api.trusted_curatorA import TrustedCurator
    from api.policyA import Policy
    from api.models import DNN_CV, OurDataset, methodology2
except:
    from project2.api.users import credentials
    from project2.api.trusted_curatorA import TrustedCurator
    from project2.api.policyA import Policy
    from project2.api.models import DNN_CV, OurDataset, methodology2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# call trusted curator.
tc = TrustedCurator(
    user='master',
    password='123456789',
    mode='off',
)

# call policy.
pl = Policy(
    n_actions=3,
    action_set=['Vaccine1', 'Vaccine2', 'Vaccine3'],
)

# get 1st sample of population already vaccinated.
X = tc.get_features(
    n_population=10000
)

A = pl.get_actions(features=X)

Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

X = tc.get_features(
    n_population=10000
)

A = pl.get_actions(features=X)

Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

A = pl.get_actions(features=X)

Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

A = pl.get_actions(features=X)

Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

A = pl.get_actions(features=X)

Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

# error: File "mtrand.pyx", line 954, in numpy.random.mtrand.RandomState.choice
# ValueError: Cannot take a larger sample than population when 'replace=False'
# policy.py line 114

vaccines = ["Vaccine1", "Vaccine2", "Vaccine3"]
symptoms = ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell', 'Fever',
            'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots']
full_training_idxs = ['Age', 'Gender', 'Income'] + \
                     [f'g{i}' for i in range(1, 128+1)] + \
                     ['Asthma', 'Obesity', 'Smoking', 'Diabetes',
                      'Heart-disease', 'Hypertension'] + \
                     [f'Vaccine{i}' for i in range(1, 3+1)]

# ~Vaccine1 and ~Vaccine2 and ~Vaccine3
# goal: minimize the number of death between vaccinated people
not_vaccinated = X[X['Vaccine1'] == 0]
not_vaccinated = not_vaccinated[not_vaccinated['Vaccine2'] == 0]
not_vaccinated = not_vaccinated[not_vaccinated['Vaccine3'] == 0]

# Vaccine1 or Vaccine2 or Vaccine3
# goal: use this data to get a policy under an utility
vaccinated = X[X['Vaccine1'] == 1]
vaccinated = pd.concat([vaccinated, X[X['Vaccine2'] == 1]], axis=0)
vaccinated = pd.concat([vaccinated, X[X['Vaccine3'] == 1]], axis=0)

# dead individuals after vaccination
dead_individuals = vaccinated[vaccinated['Death']==1]
dead_individuals = dead_individuals[dead_individuals['Covid-Positive']==1]

# sensitivity study:
parameter = "Death"
num_top_features = 10
seed = 1
n_jobs = -1

real_base = vaccinated.drop(
    ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell', 'Fever',
     'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots'],
    axis=1
)

real_base_corr = real_base.corr()
real_base_neg_corr = real_base_corr[parameter].sort_values().head(30)
real_base_pos_corr = real_base_corr[parameter].sort_values().tail(30)

top_pos = real_base_pos_corr.index[(-1-num_top_features):-1].to_list()
top_neg = real_base_neg_corr.index[:num_top_features].to_list()

relevant_features = top_neg + top_pos + ["Death"]

for v in vaccines:
    if v not in relevant_features:
        relevant_features += [v]

real_base = real_base[relevant_features]

df_dead = real_base[real_base[parameter] == 1.0]
df_not_dead = real_base[real_base[parameter] == 0.0]
df_not_dead = df_not_dead.sample(
    n=df_dead.shape[0],
    replace=False,
    random_state=seed,
)
df_balanced = pd.concat([df_dead, df_not_dead])

steps = [
    ('lr',
     LogisticRegression(
         solver='saga',
         random_state=seed,
         n_jobs=n_jobs,
         verbose=False
     ))
]
pipeline = Pipeline(steps)

param_dist = {
    'lr__penalty': ['l2', 'l1'],
    'lr__max_iter': [5, 9, 10, 15, 20, 25, 50, 100],
    'lr__C': [0.1, 0.25, 0.5, 0.75, 1]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=100,
    scoring='accuracy',
    refit=True,
    cv=5,
    random_state=seed
)

training_idxs = df_balanced.drop(["Death"], axis=1).columns.to_list()
search.fit(df_balanced[training_idxs], df_balanced["Death"])
print(f'Best Cross-Validated mean score: {search.best_score_}')
model = search.best_estimator_

X_pred = not_vaccinated[training_idxs]

actions_true = np.ones(shape=(X_pred.shape[0], 1))
actions_false = np.zeros(shape=(X_pred.shape[0], 1))

steps = [
    [actions_false, actions_false, actions_false],
    [actions_true, actions_false, actions_false],
    [actions_false, actions_true, actions_false],
    [actions_false, actions_false, actions_true],
]

saved_pred = []
saved_pob = []

for s in steps:
    X_pred['Vaccine1'] = s[0]
    X_pred['Vaccine2'] = s[1]
    X_pred['Vaccine3'] = s[2]

    saved_pred.append(model.predict(X_pred))
    saved_pob.append(model.predict_proba(X_pred))

# continue here !!
p_tilde_a1 = sum(saved_pred[1])/len(saved_pred[1])
p_tilde_a2 = sum(saved_pred[2])/len(saved_pred[2])
p_tilde_a3 = sum(saved_pred[3])/len(saved_pred[3])

# Utility := Expected estimation of the number of deaths
# num of D | A_1
da1 = sum(vaccinated[vaccinated["Vaccine1"]==1]["Death"])
p_hat_a1 = da1/vaccinated.shape[0]
expected_utility_a1 = vaccinated.shape[0] * p_hat_a1

# num of D | A_2
da2 = sum(vaccinated[vaccinated["Vaccine2"]==1]["Death"])
p_hat_a2 = da2/vaccinated.shape[0]
expected_utility_a2 = vaccinated.shape[0] * p_hat_a2

# num of D | A_3
da3 = sum(vaccinated[vaccinated["Vaccine3"]==1]["Death"])
p_hat_a3 = da3/vaccinated.shape[0]
expected_utility_a3 = vaccinated.shape[0] * p_hat_a3

# num of D | A_1 or A_2 or A_3
y = sum(vaccinated["Death"])
p_hat = y/vaccinated.shape[0]
expected_utility = vaccinated.shape[0] * p_hat

# LR decision
for indx, indv in enumerate(not_vaccinated.index):
    decision = np.argmax(
        np.array([
            saved_pob[0][indx][0],
            saved_pob[1][indx][0],
            saved_pob[2][indx][0],
            saved_pob[3][indx][0]
            ])
    )
    if decision == 1:
        not_vaccinated["Vaccine1"][indv] = 1
    elif decision == 2:
        not_vaccinated["Vaccine2"][indv] = 1
    elif decision == 3:
        not_vaccinated["Vaccine3"][indv] = 1

# apply the decided action to the individuals.
A = not_vaccinated.iloc[:, -3:]
Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)
Y.index = A.index.to_list()

# num of D | A_1
da1 = sum(Y[A["Vaccine1"]==1]["Death"])
p_prime_a1 = da1/Y.shape[0]
expected_utility_prime_a1 = Y.shape[0] * p_prime_a1

# num of D | A_2
da2 = sum(Y[A["Vaccine2"]==1]["Death"])
p_prime_a2 = da2/Y.shape[0]
expected_utility_prime_a2 = Y.shape[0] * p_prime_a2

# num of D | A_3
da3 = sum(Y[A["Vaccine3"]==1]["Death"])
p_prime_a3 = da3/Y.shape[0]
expected_utility_prime_a3 = Y.shape[0] * p_prime_a3

# num of D | A_1 or A_2 or A_3
da = sum(Y["Death"])
p_prime_a = da/Y.shape[0]
expected_utility_prime_a = Y.shape[0] * p_prime_a

# random decision: for 1 round
for indv in not_vaccinated.index:
    not_vaccinated[np.random.choice(vaccines)][indv] = 1

# apply the decided action to the individuals.
A = not_vaccinated.iloc[:, -3:]
Y = tc.get_outcomes(
    individuals_idxs=A.index.to_list(),
    actions=A,
)

# Only dead individuals Covid-Positive and Vaccinated
idx = A.index.to_list()
Y.index = idx
X_ = not_vaccinated[not_vaccinated.index.isin(idx)]
X_ = X_.drop(symptoms + ["Death"] + vaccines, axis=1)  # dropping old features
Y_filtered = pd.concat([Y, X_, A], axis=1)
Y_filtered = Y_filtered[Y_filtered["Covid-Positive"] == 1]

# num of D | A_1
da1 = sum(Y_filtered[Y_filtered["Vaccine1"] == 1]["Death"])
p_prime_a1 = da1/Y.shape[0]
expected_utility_prime_a1 = Y.shape[0] * p_prime_a1

# num of D | A_2
da2 = sum(Y_filtered[Y_filtered["Vaccine2"] == 1]["Death"])
p_prime_a2 = da2/Y.shape[0]
expected_utility_prime_a2 = Y.shape[0] * p_prime_a2

# num of D | A_3
da3 = sum(Y_filtered[Y_filtered["Vaccine3"] == 1]["Death"])
p_prime_a3 = da3/Y.shape[0]
expected_utility_prime_a3 = Y.shape[0] * p_prime_a3

# num of D | A_1 or A_2 or A_3
da = sum(Y_filtered["Death"])
p_prime_a = da/Y_filtered.shape[0]
expected_utility_prime_a = Y.shape[0] * p_prime_a

# train model
saved_dead_individuals = pd.concat(
    [Y_filtered, dead_individuals],
    axis=0
)

df_dead = saved_dead_individuals[saved_dead_individuals[parameter] == 1.0]
df_not_dead = saved_dead_individuals[saved_dead_individuals[parameter] == 0.0]
df_not_dead = df_not_dead.sample(
    n=df_dead.shape[0],
    replace=False,
    random_state=seed,
)
df_balanced = pd.concat([df_dead, df_not_dead])

# sensitivity study
real_base_corr = df_balanced.drop(symptoms, axis=1).corr()
real_base_neg_corr = real_base_corr[parameter].sort_values().head(30)
real_base_pos_corr = real_base_corr[parameter].sort_values().tail(30)

top_pos = real_base_pos_corr.index[(-1-num_top_features):-1].to_list()
top_neg = real_base_neg_corr.index[:num_top_features].to_list()

relevant_features = top_neg + top_pos

for v in vaccines:
    if v not in relevant_features:
        relevant_features += [v]

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
    n_iter=100,
    scoring="accuracy",
    refit=True,
    cv=5,
    random_state=seed
)

search.fit(df_balanced[relevant_features], df_balanced["Death"])
print(f'Best Cross-Validated mean score: {search.best_score_}')

# get 2nd sample of population already vaccinated.
X = tc.get_features(
    n_population=10000
)

