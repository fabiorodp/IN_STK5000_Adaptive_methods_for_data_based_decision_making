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

from sklearn.utils import resample
import statsmodels.api as sm
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
        plot_heatmap_corr(
            features=omega_3.space[syn_pos_corr.index.to_list()[-10:]].corr(),
            title='Top positive auto-correlation among selected features'
        )
        plot_heatmap_corr(
            features=omega_3.space[syn_neg_corr.index.to_list()[:10]].corr(),
            title='Top negative auto-correlation among selected features'
        )

        syn_base = omega_3.space[omega_3.space['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)
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
            data=omega_3.space,
            explanatories=syn_neg_corr[:5].keys(),
            responses=['Death']
        )

        # ############################## Methodology 3
        print('Performing methodology 3...')
        syn_df_balanced = balance_data(data=omega_3.space)

        syn_model = methodology3(
            X=syn_df_balanced[syn_top5_neg + syn_top5_pos],
            Y=syn_df_balanced['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=syn_model.best_estimator_._final_estimator,
            topn=5
        )

        syn_log_reg = sm.Logit(
            syn_df_balanced['Death'],
            syn_df_balanced[syn_top5_neg+syn_top5_pos]
        ).fit(maxiter=syn_model.best_estimator_._final_estimator.max_iter)
        print(syn_log_reg.summary())
    print('Synthetic data study completed for task 2a.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()

    treatment_base = pd.concat(
        [treatment_outcome, treatment_features.iloc[:, 10:], treatment_action],
        axis=1
    ).drop(['Covid-Positive'], axis=1)
    print('done.')

    input_ = input("Run methodologies 1, 2 and 3 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        real_neg_corr, real_pos_corr = methodology1(data=treatment_base)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)
        plot_heatmap_corr(
            features=treatment_base[real_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        real_top5_pos = real_pos_corr.index[-6:-1].to_list()
        real_top5_neg = real_neg_corr.index[:5].to_list()

        print('Performing methodology 2...')
        methodology2(
            data=treatment_base,
            explanatories=real_neg_corr[:20].keys(),
            responses=['Death']
        )

        # ############################## Methodology 3
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
            X=real_df_bootstraped[real_top5_neg + real_top5_pos],
            Y=real_df_bootstraped['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=5,
        )

        # The LR have a cross-validated rate of predicting death at 95%,
        # higher than the random guess of 50%, which confirms that the
        # top 5 positive and negative correlated features with death are
        # the most important.

        real_log_reg = sm.Logit(
            real_df_bootstraped['Death'],
            real_df_bootstraped[real_top5_neg+real_top5_pos]
        ).fit(maxiter=real_model.best_estimator_._final_estimator.max_iter)
        print(real_log_reg.summary())

        # All coefficients of the selected features are inside the confidence
        # interval with statistical significance of 5%, which ensure even
        # more that these features are the most important when predicting
        # death.

        # We are 95% confident that the coefficients are between lower
        # and higher boundaries.

    print('Real data study completed for task 2a.')
