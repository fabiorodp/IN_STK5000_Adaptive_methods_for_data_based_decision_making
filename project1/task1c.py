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

syn_save_most_corr_sympt = []
syn_save_best_model = []
symptms_labels = ['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                  'Stomach', 'Myocarditis', 'Blood-Clots']

for v in ['Vaccine1', 'Vaccine2', 'Vaccine3']:
    syn_neg_corr, syn_pos_corr = methodology1(data=omega_2.space, parameter=v)
    syn_save_most_corr_sympt.append(syn_pos_corr.keys()[-2])

    methodology2(
        data=omega_2.space,
        explanatories=[v],
        responses=[syn_pos_corr.keys()[-2]]
    )

    syn_model = methodology3(
        X=omega_2.space.drop(symptms_labels, axis=1),
        Y=omega_2.space[symptms_labels],
        max_iter=20,
        cv=10,
        seed=1,
        n_jobs=-1
    )

    syn_save_best_model.append(syn_model)

syn_best_model_v1 = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.75,
    max_iter=10
)

syn_feature_importances_results_v1 = feature_importances(
    estimator=syn_best_model_v1,
    X=syn_df_balanced.drop(['Death'], axis=1),
    y=syn_df_balanced['Death'],
    relative=False,
)


