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

eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
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

colors = ["black", "pink", "blue", "orange", "green", "yellow", "red"]
for c, ML, ov, e in zip(
        colors,
        np.vstack((saved_ML_expected_utility[4],
                   saved_ML_expected_utility[5],
                   saved_ML_expected_utility[6],
                   saved_ML_expected_utility[7],
                   saved_ML_expected_utility[8])),
        np.vstack((saved_observed_expected_utility[4],
                   saved_observed_expected_utility[5],
                   saved_observed_expected_utility[6],
                   saved_observed_expected_utility[7],
                   saved_observed_expected_utility[8])),
        np.vstack((eps[4], eps[5], eps[6], eps[7], eps[8]))
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
    [134, 113, 94, 61, 29, 39, 28, 33, 25],
    color="purple",
    marker='o',
    linestyle=':',
    linewidth=3,
    markersize=5,
    label=f"ML utility without DP"
)

plt.title("Expected utilities for ML policy given DP's epsilons")
plt.xlabel('Vaccination stages')
plt.ylabel('Estimation for the number of deaths')
plt.legend()
plt.show()
