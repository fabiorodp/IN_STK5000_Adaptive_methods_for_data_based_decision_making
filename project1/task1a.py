try:
    from .helper.generate_data import Space
    from .helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from .helper.methodologies import methodology1, methodology2, methodology3
    from .helper.help_functions import feature_importance_methodology3
    from .helper.help_functions import feature_importance_methodology1
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from project1.helper.help_functions import feature_importance_methodology3
    from project1.helper.help_functions import feature_importance_methodology1

import statsmodels.api as sm


def task1a():
    # ################################################## Synthetic data
    print('Generating synthetic data...')
    omega_1 = Space(N=100000, add_treatment=False, seed=1)
    omega_1.assign_corr_death()
    omega_1.add_correlated_symptom_with(
        explanatory_label='Covid-Positive',
        response_label='No_Taste/Smell',
        p=0.8)
    omega_1.add_correlated_symptom_with(
        explanatory_label='Covid-Positive',
        response_label='Pneumonia',
        p=0.5)
    print('done.')

    input_ = input("Run methodologies 1, 2 and 3 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        syn_neg_corr, syn_pos_corr = methodology1(data=omega_1.space)
        feature_importance_methodology1(syn_neg_corr, syn_pos_corr)
        plot_heatmap_corr(
            features=omega_1.space[syn_pos_corr.index.to_list()[-10:]].corr(),
            title='Top positive auto-correlation among selected features'
        )
        plot_heatmap_corr(
            features=omega_1.space[syn_neg_corr.index.to_list()[:10]].corr(),
            title='Top negative auto-correlation among selected features'
        )

        syn_base = omega_1.space[omega_1.space['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)
        syn_base = syn_base.drop(['Income'], axis=1)
        syn_neg_corr, syn_pos_corr = methodology1(data=syn_base)
        feature_importance_methodology1(syn_neg_corr, syn_pos_corr)
        plot_heatmap_corr(
            features=syn_base[syn_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        syn_top5_pos = syn_pos_corr.index[-6:-1].to_list()
        syn_top5_neg = syn_neg_corr.index[:5].to_list()

        # ############################## Methodology 2
        print('Performing methodology 2...')
        methodology2(
            data=omega_1.space,
            explanatories=syn_pos_corr[-10:-1].keys(),
            responses=['Death']
        )

        # ############################## Methodology 3
        print('Performing methodology 3...')
        syn_df_balanced = balance_data(data=omega_1.space)

        syn_model = methodology3(
            X=syn_df_balanced.drop(['Death', 'Income'], axis=1),
            # X=syn_df_balanced[syn_top5_neg+syn_top5_pos],
            Y=syn_df_balanced['Death'],
            max_iter=20,
            cv=500,
            seed=1,
            n_jobs=-1
        )
        # Best Cross-Validated mean score: 0.8213894736842104

        feature_importance_methodology3(
            best_model=syn_model.best_estimator_._final_estimator,
            topn=5
        )

        # The LR have a cross-validated rate of predicting death at 82%,
        # higher than the random guess of 50%, which confirms that the
        # top 5 positive and negative correlated features with death are
        # the most important.

        syn_log_reg = sm.Logit(
            syn_df_balanced['Death'],
            # syn_df_balanced[syn_top5_neg+syn_top5_pos],
            syn_df_balanced.drop(['Death', 'Income'], axis=1)
        ).fit(maxiter=syn_model.best_estimator_._final_estimator.max_iter)
        print(syn_log_reg.summary())

        # For this LR model submitted to a 500 test CV folds, with 82% in
        # predicting death, we are 95% confident that the explanatory
        # coefficients are in their Confidence interval respectively,
        # with a statistical significance level of 5%.

    print('Synthetic data study completed for task 1a.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()
    print('done...')

    input_ = input("Run methodologies 1, 2 and 3 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        real_neg_corr, real_pos_corr = methodology1(data=observation_features)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)
        plot_heatmap_corr(
            features=observation_features[real_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        real_base = observation_features[observation_features['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)
        real_neg_corr, real_pos_corr = methodology1(data=real_base)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)
        plot_heatmap_corr(
            features=real_base[real_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        real_top5_pos = real_pos_corr.index[-6:-1].to_list()
        real_top5_neg = real_neg_corr.index[:5].to_list()

        # ############################## Methodology 2
        print('Performing methodology 2...')
        methodology2(
            data=observation_features,
            explanatories=real_pos_corr[-10:-1].keys(),
            responses=['Death']
        )

        # ############################## Methodology 3
        print('Performing methodology 3...')
        real_df_balanced = balance_data(
            data=observation_features,
            param='Death'
        )

        real_model = methodology3(
            X=real_df_balanced[real_top5_neg+real_top5_pos],
            Y=real_df_balanced['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=5,
            return_top_positive=True
        )

        # The LR have a cross-validated rate of predicting death at 86%,
        # higher than the random guess of 50%, which confirms that the
        # top 5 positive and negative correlated features with death are
        # the most important.

        real_log_reg = sm.Logit(
            real_df_balanced['Death'],
            real_df_balanced[real_top5_neg+real_top5_pos]
        ).fit(maxiter=real_model.best_estimator_._final_estimator.max_iter)
        print(real_log_reg.summary())

        # All coefficients of the selected features are inside the confidence
        # interval with statistical significance of 5%, which ensure even
        # more that these features are the most important when predicting
        # death.

        # We are 95% confident that the coefficients are between lower
        # and higher boundaries.
    print('Real data study completed for task 1a.')
    # ##################################################
