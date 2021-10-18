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
import pandas as pd
from sklearn.utils import resample

# ################################################## Synthetic data
omega_3 = Space(N=100000, add_treatment=True)
omega_3.assign_corr_death()
omega_3.add_correlated_symptom_with(
    explanatory_label='Treatment1',
    response_label='Headache',
    p=0.5)
omega_3.add_correlated_symptom_with(
    explanatory_label='Treatment2',
    response_label='Fever',
    p=0.7)

syn_neg_corr, syn_pos_corr = methodology1(data=omega_3.space)

methodology2(
    data=omega_3.space,
    explanatories=syn_neg_corr[:5].keys(),
    responses=['Death']
)

syn_df_balanced = balance_data(data=omega_3.space)

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
    C=0.75,
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

treatment_base = pd.concat(
    [treatment_outcome, treatment_features.iloc[:, 10:], treatment_action],
    axis=1
)  # 10

real_neg_corr, real_pos_corr = methodology1(data=treatment_base)

methodology2(
    data=treatment_base,
    explanatories=real_neg_corr[:20].keys(),
    responses=['Death']
)

# Not enough data for predictions because we have only 8 individuals
# who died...
# Solutions:
# ## sklearn.resample
# ## train without balance and assign weighs and f1-score

real_df_dead = resample(
    treatment_base[treatment_base['Death'] == 1],
    replace=True,
    n_samples=300,
    random_state=1,
)

real_df_not_dead = resample(
    treatment_base[treatment_base['Death'] == 0],
    replace=False,
    n_samples=300,
    random_state=1,
)

real_df_bootstraped = pd.concat(
    [real_df_dead, real_df_not_dead],
    axis=0
)

real_model = methodology3(
    X=real_df_bootstraped.drop(['Death'], axis=1),
    Y=real_df_bootstraped['Death'],
    max_iter=20,
    cv=10,
    seed=1,
    n_jobs=-1
)

real_best_model = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=1,
    max_iter=5,
    penalty="l2"
)

real_feature_importances_results = feature_importances(
    real_best_model,
    real_df_bootstraped.drop(['Death'], axis=1),
    real_df_bootstraped['Death']
)
# ##################################################
