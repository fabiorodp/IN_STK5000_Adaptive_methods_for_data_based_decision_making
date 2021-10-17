try:
    from .helper.generate_data import Space
    from .helper.help_functions import balance_data, import_data
    from .helper.methodologies import methodology1, methodology2, methodology3
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import balance_data, import_data

from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import feature_importances

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
    data=omega_1.space,
    explanatories=syn_pos_corr[-10:-1].keys(),
    responses=['Death']
)

syn_df_balanced = balance_data(data=omega_1.space)

syn_model = methodology3(
    X=syn_df_balanced.drop(['Death'], axis=1),
    Y=syn_df_balanced['Death'],
    max_iter=20,
    cv=10,
    seed=1,
    n_jobs=-1
)

syn_best_model = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.1,
    max_iter=10
)

syn_feature_importances_results = feature_importances(
    estimator=syn_best_model,
    X=syn_df_balanced.drop(['Death'], axis=1),
    y=syn_df_balanced['Death'],
    relative=False,
)
# ##################################################

# ################################################## Real data
observation_features, treatment_features, \
    treatment_action, treatment_outcome = import_data()

real_neg_corr, real_pos_corr = methodology1(data=observation_features)

methodology2(
    data=observation_features,
    explanatories=real_pos_corr[-10:-1].keys(),
    responses=['Death']
)

real_df_balanced = balance_data(data=observation_features)

real_model = methodology3(
    X=real_df_balanced.drop(['Death'], axis=1),
    Y=real_df_balanced['Death'],
    max_iter=20,
    cv=10,
    seed=1,
    n_jobs=-1
)

real_best_model = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.75,
    max_iter=10,
    penalty="l2"
)

real_feature_importances_results = feature_importances(
    real_best_model,
    real_df_balanced.drop(['Death'], axis=1),
    real_df_balanced['Death']
)
# ##################################################
