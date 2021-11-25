try:
    from api.users import credentials
    from api.trusted_curator import TrustedCurator
    from api.policy import Policy
    from api.models import DNN_CV, OurDataset, methodology2
    from api.helper import fairness_barplot, fairness_lineplot
    from api.helper import plot_expected_utilities
    from api.helper import plot_dist_age, plot_dist_gender, plot_dist_income
    from api.helper import fairness_simple_probabilities
    from api.helper import fairness_calibration
    from api.helper import fairness_unifying_view
except:
    from project2.api.users import credentials
    from project2.api.trusted_curator import TrustedCurator
    from project2.api.policy import Policy
    from project2.api.models import DNN_CV, OurDataset, methodology2
    from project2.api.helper import fairness_barplot, fairness_lineplot
    from project2.api.helper import plot_expected_utilities
    from project2.api.helper import plot_dist_age, plot_dist_gender
    from project2.api.helper import plot_dist_income
    from project2.api.helper import fairness_simple_probabilities
    from project2.api.helper import fairness_calibration
    from project2.api.helper import fairness_unifying_view

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    plot_fairness=False,
    num_top_features=10,
    fairness=False
)

for _ in range(10):
    X = tc.get_features(
        n_population=10000
    )

    A = pl.get_actions(features=X)

    Y = tc.get_outcomes(
        individuals_idxs=A.index.to_list(),
        actions=A,
    )

    pl.observe(
        actions=A,
        outcomes=Y
    )

# #################### distributions
plot_dist_age(pl), plot_dist_gender(pl), plot_dist_income(pl)

# #################### printing the simple probabilities for fairness
tables_pa_z, pa = fairness_simple_probabilities(pl)

# #################### printing the calibration for fairness
tables_py_a, tables_py_az = fairness_calibration(pl)

# #################### finding | P(y|a,z) - P(y|a) |
print("finding | P(y|a,z) - P(y|a) | for each space....")
print(pd.DataFrame(abs(tables_py_az[0].values - tables_py_a[0].T.values)))
print(pd.DataFrame(abs(tables_py_az[1].values - tables_py_a[1].T.values)))
print(pd.DataFrame(abs(tables_py_az[2].values - tables_py_a[2].T.values)))

# #################### printing the unified view of fairness
# Table1 : all vaccinated population
tb1 = fairness_unifying_view(tables_py_az[0].values)

# Table 2 : vaccinated by us
tb2 = fairness_unifying_view(tables_py_az[1].values)

# Table 3 : vaccinated before
tb3 = fairness_unifying_view(tables_py_az[2].values)
