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

from sklearn.utils import resample
import pandas as pd


def task2a():
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
            explanatories=syn_neg_corr[:5].keys(),
            responses=['Death']
        )

    input_ = input("Run methodology 3 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing...')
        syn_df_balanced = balance_data(data=omega_3.space)

        syn_model = methodology3(
            X=syn_df_balanced.drop(['Death'], axis=1),
            Y=syn_df_balanced['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=syn_model.best_estimator_._final_estimator,
            topn=15
        )
    print('Synthetic data study completed for task 2a.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()

    treatment_base = pd.concat(
        [treatment_outcome, treatment_features.iloc[:, 10:], treatment_action],
        axis=1
    )
    print('done.')

    input_ = input("Run methodologies 1 and 2 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')

        real_neg_corr, real_pos_corr = methodology1(data=treatment_base)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)

        print('Performing methodology 2...')
        methodology2(
            data=treatment_base,
            explanatories=real_neg_corr[:20].keys(),
            responses=['Death']
        )
    print('done.')

    input_ = input("Run methodology 3 for real data? (y/n)\n")
    if input_ == 'y':

        real_df_dead = resample(
            treatment_base[treatment_base['Death'] == 1],
            replace=True,
            n_samples=300,
            random_state=1,
        )

        real_df_not_dead = resample(
            treatment_base[treatment_base['Death'] == 0],
            replace=False,
            n_samples=300,
            random_state=1,
        )

        real_df_bootstraped = pd.concat(
            [real_df_dead, real_df_not_dead],
            axis=0
        )

        real_model = methodology3(
            X=real_df_bootstraped.drop(['Death'], axis=1),
            Y=real_df_bootstraped['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=15
        )
    print('Real data study completed for task 2a.')
