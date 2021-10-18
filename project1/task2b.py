try:
    from .helper.generate_data import Space
    from .helper.models import mlp
    from .helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from .helper.methodologies import methodology1, methodology2, methodology3
except:
    from project1.helper.generate_data import Space
    from project1.helper.methodologies import methodology1, methodology2, methodology3
    from project1.helper.help_functions import balance_data, import_data, plot_heatmap_corr
    from project1.helper.models import mlp

from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import feature_importances
import pandas as pd
from sklearn.utils import resample
from seaborn import heatmap
from matplotlib.pyplot import show
import tensorflow as tf
from sklearn.model_selection import KFold

# ################################################## Synthetic data
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

syn_neg_corr, syn_pos_corr = methodology1(data=omega_3.space)

methodology2(
    data=omega_3.space,
    explanatories=['Treatment1', 'Treatment2'],
    responses=['Headache', 'Fever']
)

syn_save_df_balance = []
syn_save_best_model = []
for s in ['Headache', 'Fever']:
    syn_df_balanced = balance_data(
        data=omega_3.space,
        param=s
    )
    syn_save_df_balance.append(syn_df_balanced)

    syn_model = methodology3(
        X=syn_df_balanced.drop([s], axis=1),
        Y=syn_df_balanced[s],
        max_iter=20,
        cv=10,
        seed=1,
        n_jobs=-1
    )
    syn_save_best_model.append(syn_model)

# best model for 'Headache'
syn_best_model_headache = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.75,
    max_iter=10
)

syn_feature_importances_results_headache = feature_importances(
    estimator=syn_best_model_headache,
    X=syn_save_df_balance[0].drop(['Headache'], axis=1),
    y=syn_save_df_balance[0]['Headache'],
    relative=False,
)

# best model for 'Fever'
syn_best_model_fever = LogisticRegression(
    solver='saga',
    random_state=1,
    n_jobs=-1,
    C=0.75,
    max_iter=10
)

syn_feature_importances_results_fever = feature_importances(
    estimator=syn_best_model_fever,
    X=syn_save_df_balance[1].drop(['Fever'], axis=1),
    y=syn_save_df_balance[1]['Fever'],
    relative=False,
)

# ##################################################

# ################################################## Real data
observation_features, treatment_features, \
    treatment_action, treatment_outcome = import_data()

treatment_base_before = pd.concat(
    [treatment_features, treatment_action],
    axis=1
)

treatment_base_after = pd.concat(
    [treatment_outcome, treatment_features.iloc[:, 11:], treatment_action],
    axis=1
)

real_neg_corr, real_pos_corr = methodology1(data=treatment_base_after)

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

kfold = KFold(n_splits=10, shuffle=True)
inputs = treatment_base_after.iloc[:, 10:]
targets = treatment_base_after.iloc[:, 2:9]
history, scores = [], []
for train, test in kfold.split(inputs, targets):
    model = mlp(
        input_shape=treatment_base_after.iloc[:, 10:].shape,
        num_of_hidden_layers=2,
        hls_units=[100, 100],
        hls_act_functs=[tf.nn.relu, tf.nn.relu],
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        output_shape=7,
        output_act_funct=tf.nn.softmax,
        dropout=0.2
    )

    history.append(model.fit(
        inputs.iloc[train, :],
        targets.iloc[train, :],
        # batch_size=32,
        epochs=10,
        verbose=1
    ))

    scores.append(
        model.evaluate(
            inputs.iloc[test],
            targets.iloc[test],
            verbose=0
        )
    )
