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
    num_top_features=10,
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

"""
import pickle

with open('pl.pkl', 'wb') as outp:
    pickle.dump(pl, outp, pickle.HIGHEST_PROTOCOL)

with open('project2/csv/pl.pkl', 'rb') as inp:
    pl = pickle.load(inp)
"""

# #################### historical data
vaccinated_by_us, vaccinated_before, all_pop_vaccinated = pl.vstack_all()

# #################### 1)
estimated_utility_vb = []
estimated_utility_vu = []
estimated_utility_all = []

for df, outlist in zip(
    [vaccinated_before, vaccinated_by_us, all_pop_vaccinated],
    [estimated_utility_vb, estimated_utility_vu, estimated_utility_all],
):
    for i in range(1000):
        sample_data = df.sample(frac=0.25, replace=True)
        prob = sample_data['Death'].sum()[0] / sample_data.shape[0]
        outlist.append(prob)

df_to_plot = pd.DataFrame({
    "Vaccinated before": estimated_utility_vb,
    "Vaccinated by us": estimated_utility_vu,
    "All population vaccinated": estimated_utility_all
})

sns.boxplot(
    data=df_to_plot,
    orient='h'
).figure.subplots_adjust(left=0.35, bottom=0.15)
plt.title(f'Historical policy utility estimation')
plt.xlabel("Utility: Probability of death")
plt.ylabel("Sample spaces")
plt.show()

df_to_plot.describe()

# real utilities
utility_vb = vaccinated_before['Death'].sum()[0] / vaccinated_before.shape[0]
utility_vu = vaccinated_by_us['Death'].sum()[0] / vaccinated_by_us.shape[0]
utility_all = \
    all_pop_vaccinated['Death'].sum()[0] / all_pop_vaccinated.shape[0]

df_to_table = pd.DataFrame({
    "Vaccinated before": [utility_vb],
    "Vaccinated by us": [utility_vu],
    "All population vaccinated": [utility_all]
})

print(df_to_table)
