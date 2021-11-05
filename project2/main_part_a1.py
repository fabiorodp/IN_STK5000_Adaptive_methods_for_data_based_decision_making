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

plt.plot(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    pl.ML_expected_utilities,
    color='green',
    marker='o',
    linestyle='dashed',
    linewidth=2,
    markersize=5,
    label="ML Policy"
)
plt.plot(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [np.mean(pl.random_expected_utilities) for _ in range(9)],
    color='red',
    marker='o',
    linestyle='dashed',
    linewidth=2,
    markersize=5,
    label="Mean for Random Policy"
)
plt.plot(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    pl.observed_expected_utilities,
    color='orange',
    marker='o',
    linestyle='dashed',
    linewidth=2,
    markersize=5,
    label="Observed Deaths"
)
plt.title('Expected utilities for ML and Random vaccination policies')
plt.xlabel('Vaccination stages')
plt.ylabel('Estimation for the number of deaths')
plt.legend()
plt.show()
