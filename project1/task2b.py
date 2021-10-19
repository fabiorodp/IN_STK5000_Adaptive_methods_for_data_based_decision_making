try:
    from .helper.generate_data import Space
    from .helper.help_functions import balance_data, import_data
    from .helper.methodologies import methodology1, methodology2, methodology3
    from .helper.help_functions import feature_importance_methodology3
    from .helper.help_functions import feature_importance_methodology1
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import balance_data, import_data
    from project1.helper.help_functions import feature_importance_methodology3
    from project1.helper.help_functions import feature_importance_methodology1

import pandas as pd


def task2b():
    # ################################################## Synthetic data
    print('Generating synthetic data...')
    omega_3 = Space(N=100000, add_treatment=True)
    omega_3.assign_corr_death()
    omega_3.add_correlated_symptom_with(
        explanatory_label='Treatment1',
        response_label='Headache',
        p=0.5)
    omega_3.add_correlated_symptom_with(
        explanatory_label='Treatment2',
        response_label='Fever',
        p=0.7)
    print('done.')

    input_ = input("Run methodologies 1 and 2 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')

        syn_neg_corr, syn_pos_corr = methodology1(data=omega_3.space)
        feature_importance_methodology1(syn_neg_corr, syn_pos_corr)

        print('Performing methodology 2...')
        methodology2(
            data=omega_3.space,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['Headache', 'Fever']
        )
    print('Synthetic data study completed for task 2b.')


    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()

    treatment_base_before = pd.concat(
        [treatment_features, treatment_action],
        axis=1
    )

    treatment_base_after = pd.concat(
        [treatment_outcome, treatment_features.iloc[:, 10:], treatment_action],
        axis=1
    )
    print('done.')

    input_ = input("Run methodologies 1 and 2 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        real_neg_corr, real_pos_corr = methodology1(data=treatment_base_after)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)

        print('Methodology 2 - Before...\n\n')
        methodology2(
            data=treatment_base_before,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )

        print('Methodology 2 - After...\n\n')
        methodology2(
            data=treatment_base_after,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )
    print('Real data study completed for task 2b.')

    # ##################################################
