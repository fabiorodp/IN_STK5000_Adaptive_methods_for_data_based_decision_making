from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pipelinehelper import PipelineHelper
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

try:
    from helper.help_functions import import_data, plot_heatmap_corr
    from helper.help_functions import age_analysis

except:
    from project1.helper.help_functions import import_data, plot_heatmap_corr
    from project1.helper.help_functions import age_analysis

seed = 1
n_jobs = -1

# importing data
(observation_features, treatment_features, treatment_action,
 treatment_outcome) = import_data()

# separating infected and not
infected = observation_features[observation_features[1] == 1.].drop(
    [1], axis=1)
not_infected = observation_features[observation_features[1] == 0.].drop(
    [1], axis=1)

# #################### correlation analysis

# among the symptoms: No-Taste/Smell, Fever, Headache, Pneumonia, Stomach,
# Myocarditis, Blood-Clots, Death.
plot_heatmap_corr(
    df=infected[[2, 3, 4, 5, 6, 7, 8, 9]],
    labels=['No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
            'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'],
    _show=True
)
# not a significant correlation among symptoms

# among genes: 128 bits
genes_corr = infected.iloc[:, 12:140].corr()

# is there any correlation above 5% ?
((genes_corr > 0.0001) & (genes_corr < 1.0)).sum()

# #################### How many people died?
infected[9].value_counts()
# 0.0    21717
# 1.0      311   --> only 311 people??

# #################### age analysis
age_analysis(df=infected, plot_box=True, plot_dist=True)

# #################### gender analysis

# How many of death per gender?

# age given death & female & male
infected[10][(infected[9] == 1.0)].plot(
    kind="box",
    label="age given death & female & male"
)
plt.show()

# age given death & female
infected[10][(infected[9] == 1.0) & (infected[11] == 0.)].plot(
    kind="box",
    label="age given death & female",
)
plt.show()

# age given death & male
infected[10][(infected[9] == 1.0) & (infected[11] == 1.)].plot.box(
    label="age given death & male",
)
plt.show()
