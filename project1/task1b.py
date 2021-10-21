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

    input_ = input("Run methodologies 1, 2 and 3 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        syn_base = omega_2.space[omega_2.space['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)
        syn_base = syn_base.drop(['Income'], axis=1)
        syn_neg_corr, syn_pos_corr = methodology1(data=syn_base)
        feature_importance_methodology1(syn_neg_corr, syn_pos_corr)
        plot_heatmap_corr(
            features=syn_base[syn_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        syn_topp = syn_pos_corr.index[-11:-1].to_list()
        syn_topn = syn_neg_corr.index[:10].to_list()

        # ############################## Methodology 2
        print('Performing methodology 2...')
        methodology2(
            data=syn_base,
            explanatories=syn_neg_corr[:5].keys(),
            responses=['Death']
        )

        # ############################## Methodology 3
        print('Performing methodology 3...')
        syn_df_balanced = balance_data(data=omega_2.space)

        syn_model = methodology3(
            X=syn_df_balanced[syn_topn + syn_topp],
            Y=syn_df_balanced['Death'],
            max_iter=20,
            cv=500,
            seed=1,
            n_jobs=-1
        )
        # Best Cross-Validated mean score: 0.8252116253691499

        feature_importance_methodology3(
            best_model=syn_model.best_estimator_._final_estimator,
            topn=5
        )

        syn_log_reg = sm.Logit(
            syn_df_balanced['Death'],
            syn_df_balanced[syn_topn+syn_topp]
        ).fit(maxiter=syn_model.best_estimator_._final_estimator.max_iter)
        print(syn_log_reg.summary())

        confidence_interval_plot(
            lr_model=syn_log_reg,
            top=5
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
        real_base = observation_features[
            observation_features['Covid-Positive'] == 1].drop(
            ['Covid-Positive'], axis=1)
        real_neg_corr, real_pos_corr = methodology1(data=real_base)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)
        plot_heatmap_corr(
            features=real_base[real_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        real_topp = real_pos_corr.index[-11:-1].to_list()
        real_topn = real_neg_corr.index[:10].to_list()

        # ############################## Methodology 2
        print('Performing methodology 2...')
        methodology2(
            data=observation_features,
            explanatories=real_neg_corr[:5].keys(),
            responses=['Death']
        )
        print('done.')

        # ############################## Methodology 3
        print('Performing methodology 3...')
        real_df_balanced = balance_data(
            data=observation_features,
            param='Death'
        )

        real_model = methodology3(
            X=real_df_balanced[real_topn+real_topp],
            Y=real_df_balanced['Death'],
            max_iter=20,
            cv=500,
            seed=1,
            n_jobs=-1
        )

        # Best Cross-Validated mean score: 0.8700404858299594

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=5,
            return_top_positive=True
        )

        syn_log_reg = sm.Logit(
            real_df_balanced['Death'],
            real_df_balanced[real_topn+real_topp]
        ).fit(maxiter=real_model.best_estimator_._final_estimator.max_iter)
        print(syn_log_reg.summary())

        confidence_interval_plot(
            lr_model=syn_log_reg,
            top=5
        )
    print('Real data study completed for task 1b.')
    # ##################################################
