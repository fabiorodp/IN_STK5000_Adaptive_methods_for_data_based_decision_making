try:
    from .helper.generate_data import Space
    from .helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from .helper.help_functions import confidence_interval_plot
    from .helper.methodologies import methodology1, methodology2, methodology3
    from .helper.help_functions import feature_importance_methodology3
    from .helper.help_functions import feature_importance_methodology1
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import confidence_interval_plot
    from project1.helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from project1.helper.help_functions import feature_importance_methodology3
    from project1.helper.help_functions import feature_importance_methodology1

import statsmodels.api as sm


def task1c():
    # ################################################## Synthetic data
    print('Generating synthetic data...')
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
    print('done.')

    input_ = input("Run methodologies 2 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 2...')
        methodology2(
            data=omega_2.space,
            explanatories=['Vaccine1', 'Vaccine2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )
    print('Synthetic data study completed for task 1c.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()
    print('done.')

    input_ = input("Run methodologies 2 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 2...')
        methodology2(
            data=observation_features,
            explanatories=['Vaccine1', 'Vaccine2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )
    print('Real data study completed for task 1c.')

    # ##################################################
