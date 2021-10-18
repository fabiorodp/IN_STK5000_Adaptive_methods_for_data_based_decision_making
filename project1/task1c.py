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

# ################################################## Synthetic data
omega_2 = Space(N=100000, add_treatment=False, seed=1)
omega_2.assign_corr_death()
omega_2.add_correlated_symptom_with(
    explanatory_label='Vaccine1',
    response_label='Blood-Clots',
    p=0.3)
omega_2.add_correlated_symptom_with(
    explanatory_label='Vaccine2',
    response_label='Headache',
    p=0.6)
omega_2.add_correlated_symptom_with(
    explanatory_label='Vaccine3',
    response_label='Fever',
    p=0.7)

syn_neg_corr, syn_pos_corr = methodology1(data=omega_2.space)

methodology2(
    data=omega_2.space,
    explanatories=['Vaccine1', 'Vaccine2'],
    responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
               'Stomach', 'Myocarditis', 'Blood-Clots']
)

# ##################################################

# ################################################## Real data
observation_features, treatment_features, \
    treatment_action, treatment_outcome = import_data()

real_neg_corr, real_pos_corr = methodology1(data=observation_features)

methodology2(
    data=observation_features,
    explanatories=['Vaccine1', 'Vaccine2'],
    responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
               'Stomach', 'Myocarditis', 'Blood-Clots']
)

# ##################################################
