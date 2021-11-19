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

# #################### printing the unified view of fairness
# Table1 : all vaccinated population
"""r1 = [0.004521, 0.004092, 0.004595, 0.004377, 0.004340, 0.004173, 0.004686]
r2 = [0.004246, 0.005750, 0.004181, 0.004646, 0.005028, 0.004628, 0.005194]
r3 = [0.005253, 0.005862, 0.004021, 0.005237, 0.005521, 0.005662, 0.004898]
tb1 = np.array([r1, r2, r3])
tb1 = fairness_unifying_view(tb1)
print(np.max(tb1))"""
tb1 = fairness_unifying_view(tables_py_az[0].values)


# Table 2 : vaccinated by us
"""r1 = [0.004801, 0.004196, 0.004317, 0.005258, 0.003786, 0.004224, 0.005059]
r2 = [0.004057, 0.006237, 0.004082, 0.004546, 0.005310, 0.004422, 0.005788]
r3 = [0.005608, 0.005802, 0.005443, 0.005421, 0.005918, 0.005931, 0.005231]
tb2 = np.array([r1, r2, r3])
tb2 = fairness_unifying_view(tb2)
print(np.max(tb2))"""
tb2 = fairness_unifying_view(tables_py_az[1].values)

# Table 3 : vaccinated before
"""r1 = [0.003806, 0.003837, 0.005198, 0.002217, 0.005712, 0.004044, 0.003803]
r2 = [0.004683, 0.004595, 0.004425, 0.004882, 0.004374, 0.005110, 0.003800]
r3 = [0.004455, 0.006002, 0.001038, 0.004821, 0.004618, 0.005061, 0.004127]
tb3 = np.array([r1, r2, r3])
tb3 = fairness_unifying_view(tb3)
print(np.max(tb3))"""
tb3 = fairness_unifying_view(tables_py_az[2].values)
