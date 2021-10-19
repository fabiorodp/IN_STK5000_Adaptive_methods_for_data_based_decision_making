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


def task1b():
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

    input_ = input("Run methodologies 1 and 2 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        syn_neg_corr, syn_pos_corr = methodology1(data=omega_2.space)
        feature_importance_methodology1(syn_neg_corr, syn_pos_corr)

        print('Performing methodology 2...')
        methodology2(
            data=omega_2.space,
            explanatories=syn_neg_corr[:9].keys(),
            responses=['Death']
        )
        print('done.')

    input_ = input("Run methodology 3 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing...')
        syn_df_balanced = balance_data(data=omega_2.space)

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
    print('Synthetic data study completed for task 1b.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()
    print('done...')

    input_ = input("Run methodologies 1 and 2 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        real_neg_corr, real_pos_corr = methodology1(data=observation_features)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)

        print('Performing methodology 2...')
        methodology2(
            data=observation_features,
            explanatories=real_neg_corr[:9].keys(),
            responses=['Death']
        )

        print('done.')

    input_ = input("Run methodology 3 for real data? (y/n)\n")
    if input_ == 'y':
        real_df_balanced = balance_data(data=observation_features)

        real_model = methodology3(
            X=real_df_balanced.drop(['Death'], axis=1),
            Y=real_df_balanced['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=15
        )
    print('Real data study completed for task 1b.')
    # ##################################################
