try:
    from api.users import credentials
    from api.trusted_curator import TrustedCurator
    from api.policy import Policy
    from api.models import DNN_CV, OurDataset, methodology2
except:
    from project2.api.users import credentials
    from project2.api.trusted_curator import TrustedCurator
    from project2.api.policy import Policy
    from project2.api.models import DNN_CV, OurDataset, methodology2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

eps = np.arange(0.5, 20, 2)
saved_ML_expected_utility = []
saved_observed_expected_utility = []

for _, ep in zip(range(10), eps):
    # call trusted curator.
    tc = TrustedCurator(
        user='user1',
        password='12345eu',
        epsilon=ep
    )

    # call policy.
    pl = Policy(
        n_actions=3,
        action_set=['Vaccine1', 'Vaccine2', 'Vaccine3'],
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

    saved_ML_expected_utility.append(pl.ML_expected_utilities)
    saved_observed_expected_utility.append(pl.observed_expected_utilities)

colors = ["black", "pink", "blue", "orange", "yellow", "green", "red"]
for c, ML, ov, e in zip(
        colors,
        saved_ML_expected_utility[:-4],
        saved_observed_expected_utility[:-4],
        eps[:-4]
):

    plt.plot(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ML,
        color=c,
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=5,
        label=f"ML utility given epsilon {e}"
    )

plt.plot(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [140, 90, 55, 45, 80, 38, 30, 30, 25],
    color="purple",
    marker='o',
    linestyle='solid',
    linewidth=4,
    markersize=5,
    label=f"ML utility without DP"
)

plt.title("Expected utilities for ML policy given DP's epsilons")
plt.xlabel('Vaccination stages')
plt.ylabel('Estimation for the number of deaths')
plt.legend()
plt.show()
