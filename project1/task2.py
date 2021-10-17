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
omega_3 = Space(N=100000, add_treatment=True)
omega_3.add_correlated_symptom_with(
    explanatory_label='Treatment1',
    response_label='Death',
    p=0.1)
omega_3.add_correlated_symptom_with(
    explanatory_label='Treatment1',
    response_label='Headache',
    p=0.5)
omega_3.add_correlated_symptom_with(
    explanatory_label='Treatment2',
    response_label='Fever',
    p=0.7)

syn_neg_corr, syn_pos_corr = methodology1(data=omega_3.space)

# ##################################################
