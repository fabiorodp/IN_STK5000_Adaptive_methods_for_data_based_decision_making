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

from sklearn.utils import resample
import statsmodels.api as sm
import pandas as pd


def task2():
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

    input_ = input("Run methodologies 1, 2 and 3 for synthetic data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        syn_base = omega_3.space[omega_3.space['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)
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
        print('Performing methodology 2a...')
        methodology2(
            data=syn_base,
            explanatories=syn_neg_corr[:5].keys(),
            responses=['Death']
        )

        print('Performing methodology 2b...')
        methodology2(
            data=syn_base,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )

        # ############################## Methodology 3
        print('Performing methodology 3...')
        syn_df_balanced = balance_data(data=omega_3.space)

        syn_model = methodology3(
            X=syn_df_balanced[syn_topn + syn_topp],
            Y=syn_df_balanced['Death'],
            max_iter=20,
            cv=10,
            seed=1,
            n_jobs=-1
        )

        # Best Cross-Validated mean score: 0.8636048021625633

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
    print('Synthetic data study completed for task 2.')

    # ##################################################

    # ################################################## Real data
    print('Importing real data...')
    observation_features, treatment_features, \
        treatment_action, treatment_outcome = import_data()

    treatment_base_before = pd.concat(
        [treatment_features, treatment_action],
        axis=1
    ).drop(['Covid-Positive'], axis=1)

    treatment_base_after = pd.concat(
        [treatment_outcome, treatment_features.iloc[:, 10:], treatment_action],
        axis=1
    ).drop(['Covid-Positive'], axis=1)

    input_ = input("Run methodologies 1, 2 and 3 for real data? (y/n)\n")
    if input_ == 'y':
        print('Performing methodology 1...')
        real_neg_corr, real_pos_corr = methodology1(data=treatment_base_after)
        feature_importance_methodology1(real_neg_corr, real_pos_corr)
        plot_heatmap_corr(
            features=treatment_base_after[real_pos_corr.index.to_list()[-10:]].corr(),
            title='Auto-correlation among selected features'
        )

        # getting top negative and positive correlations
        real_top5_pos = real_pos_corr.index[-11:-1].to_list()
        real_top5_neg = real_neg_corr.index[:10].to_list()

        # ############################## Methodology 2
        print('Performing methodology 2a...')
        methodology2(
            data=treatment_base_after,
            explanatories=real_neg_corr[:10].keys(),
            responses=['Death']
        )

        print('Performing methodology 2b - before...')
        methodology2(
            data=treatment_base_before,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )

        print('Performing methodology 2b - after...')
        methodology2(
            data=treatment_base_after,
            explanatories=['Treatment1', 'Treatment2'],
            responses=['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia',
                       'Stomach', 'Myocarditis', 'Blood-Clots']
        )

        # ############################## Methodology 3
        real_df_dead = resample(
            treatment_base_after[treatment_base_after['Death'] == 1],
            replace=True,
            n_samples=300,
            random_state=1,
        )

        real_df_not_dead = resample(
            treatment_base_after[treatment_base_after['Death'] == 0],
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

        # Best Cross-Validated mean score: 0.9816666666666667

        feature_importance_methodology3(
            best_model=real_model.best_estimator_._final_estimator,
            topn=10,
        )

        real_log_reg = sm.Logit(
            real_df_bootstraped['Death'],
            real_df_bootstraped[real_top5_neg+real_top5_pos]
        ).fit(maxiter=real_model.best_estimator_._final_estimator.max_iter)
        print(real_log_reg.summary())

        confidence_interval_plot(
            lr_model=real_log_reg,
            top=10
        )

    print('Real data study completed for task 2.')
