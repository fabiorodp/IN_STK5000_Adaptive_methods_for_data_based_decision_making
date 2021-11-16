try:
    from api.users import credentials
    from api.trusted_curator import TrustedCurator
    from api.policy import Policy
    from api.models import DNN_CV, OurDataset, methodology2
    from api.helper import fairness_barplot, fairness_lineplot
    from api.helper import plot_expected_utilities
except:
    from project2.api.users import credentials
    from project2.api.trusted_curator import TrustedCurator
    from project2.api.policy import Policy
    from project2.api.models import DNN_CV, OurDataset, methodology2
    from project2.api.helper import fairness_barplot, fairness_lineplot
    from project2.api.helper import plot_expected_utilities

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

# #################### plotting expected utilities
plot_expected_utilities(pl=pl)

# #################### fairness bar plot for Age, Gender or Income
fairness_barplot(
    pl=pl,
    vaccination_stage=5,    # give here the wanted vaccination stage
    plot_for='Age'      # or Gender or Income
)

# #################### fairness line plot for Age, Gender or Income
fairness_lineplot(
    pl=pl,
    plot_for='Age'      # or Gender or Income
)
