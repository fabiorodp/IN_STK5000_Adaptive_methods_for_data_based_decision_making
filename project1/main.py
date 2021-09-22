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
except:
    from project1.helper.help_functions import import_data, plot_heatmap_corr


seed = 1
n_jobs = -1

# importing data
(observation_features, treatment_features, treatment_action,
 treatment_outcome) = import_data()

# separating infected and not
infected = observation_features[observation_features['0.0.1'] == 1.].drop(
    ['0.0.1'], axis=1)
not_infected = observation_features[observation_features['0.0.1'] == 0.].drop(
    ['0.0.1'], axis=1)

# checking correlations of the symptoms (No-Taste/Smell, Fever, Headache,
# Pneumonia, Stomach, Myocarditis, Blood-Clots, Death)
plot_heatmap_corr(
    df=infected[['0.0.2', '0.0.3', '0.0.4', '0.0.5', '0.0.6',
                 '0.0.7', '0.0.8', '0.0.9']],
    labels=['No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
            'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'],
    _show=True
)

plot_heatmap_corr(
    df=not_infected[['0.0.2', '0.0.3', '0.0.4', '0.0.5', '0.0.6',
                     '0.0.7', '0.0.8', '0.0.9']],
    labels=['No-Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
            'Stomach', 'Myocarditis', 'Blood-Clots', 'Death'],
    _show=True
)
