try:
    from .helper.generate_data import Space
    from .helper.help_functions import balance_data, import_data
    from .helper.methodologies import methodology1, methodology2, methodology3
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import balance_data, import_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ################################################## Synthetic data
omega_1 = Space(N=100000, add_treatment=False, seed=1)
omega_1.assign_corr_death()
omega_1.add_correlated_symptom_with(
    explanatory_label='Covid-Positive',
    response_label='No_Taste/Smell',
    p=0.8)
omega_1.add_correlated_symptom_with(
    explanatory_label='Covid-Positive',
    response_label='Pneumonia',
    p=0.5)

syn_neg_corr, syn_pos_corr = methodology1(data=omega_1.space)

methodology2(
    data=omega_1 .space,
    explanatories=['Age', 'g1', 'g2', 'Diabetes',
                   'Hypertension', 'Income'],
    responses=['Death', 'Pneumonia', 'No_Taste/Smell']
)

df_balanced = balance_data(data=omega_1.space)

syn_model = methodology3(
    X=df_balanced.drop(['Death'], axis=1),
    Y=df_balanced['Death'],
    max_iter=5,
    cv=10,
    seed=1,
    n_jobs=-1
)
# ##################################################

# ################################################## Real data
observation_features, treatment_features, \
    treatment_action, treatment_outcome = import_data()

real_neg_corr, real_os_corr = methodology1(data=observation_features)

methodology2(
    data=observation_features,
    explanatories=['Age', 'g1', 'g2', 'Diabetes',
                   'Hypertension', 'Income'],
    responses=['Death', 'Pneumonia', 'No-Taste/Smell']
)

df_balanced = balance_data(data=observation_features)

real_model = methodology3(
    X=df_balanced.drop(['Death'], axis=1),
    Y=df_balanced['Death'],
    max_iter=5,
    cv=10,
    seed=1,
    n_jobs=-1
)

best_model = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.1,
    max_iter=500
)

cross_validated_results = cross_val_score(
    estimator=best_model,
    X=df_balanced.drop(['Death'], axis=1),
    y=df_balanced['Death'],
    scoring='accuracy',
    cv=10,
    n_jobs=-1
)

best_model.fit(
    X=df_balanced.drop(['Death'], axis=1),
    y=df_balanced['Death'],
)

best_model_coeffs = best_model.coef_

# ##################################################
