try:
    from models import DNN_CV
except:
    from project2.api.models import DNN_CV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class Policy:
    """ A policy for treatment/vaccination. """

    def __init__(self, n_actions, action_set, seed=1, n_jobs=-1):
        """ Initialise.
        Args:
        n_actions (int): the number of actions
        action_set (list): the set of actions
        """
        self.n_actions = n_actions
        self.action_set = action_set
        print("Initialising policy with ", n_actions, "actions")
        print("A = {", action_set, "}")

        self.vaccines = [f'Vaccine{i}' for i in range(1, n_actions + 1)]

        self.symptoms = ['Covid-Recovered', 'Covid-Positive',
                         'No_Taste/Smell', 'Fever', 'Headache',
                         'Pneumonia', 'Stomach', 'Myocarditis',
                         'Blood-Clots']

        self.full_training_idxs = ['Age', 'Gender', 'Income'] + \
                                  [f'g{i}' for i in range(1, 128 + 1)] + \
                                  ['Asthma', 'Obesity', 'Smoking', 'Diabetes',
                                   'Heart-disease', 'Hypertension'] + \
                                  self.vaccines

        self.vaccination_stage = 0
        self.response_parameter = "Death"
        self.num_top_features = 10

        self.search_pipeline = None
        self.relevant_features = None

        self.seed = seed
        self.n_jobs = n_jobs

        self.not_vaccinated = None
        self.vaccinated = None
        self.old_saved_dead_individuals = None
        self.saved_dead_individuals = pd.DataFrame()

        self.observed_P_D_given_v1 = None
        self.observed_P_D_given_v1 = None
        self.observed_P_D_given_v1 = None
        self.observed_P_D_given_v123 = None

        self.expected_num_D_given_v1 = None
        self.expected_num_D_given_v2 = None
        self.expected_num_D_given_v3 = None
        self.expected_num_D_given_v123 = None

    def create_new_ml_pipeline(self):
        steps = [
            ('rf', RandomForestClassifier(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=False
            ))
        ]

        pipeline = Pipeline(steps)

        param_dist = {
            'rf__n_estimators': [25, 50, 100, 500, 1000],
            'rf__criterion': ['gini', 'entropy'],
            'rf__bootstrap': [True, False],
        }

        self.search_pipeline = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=100,
            scoring="accuracy",
            refit=True,
            cv=5,
            random_state=self.seed
        )

    def save_dead_individuals_for_observe(self, A, Y):
        idx = A.index.to_list()
        Y.index = idx
        X_ = self.not_vaccinated[self.not_vaccinated.index.isin(idx)]
        X_ = X_.drop(self.symptoms + ["Death"] + self.vaccines,
                     axis=1)  # dropping old features

        X_filtered = pd.concat([Y, X_, A], axis=1)
        X_filtered = X_filtered[X_filtered["Covid-Positive"] == 1]

        self.saved_dead_individuals = pd.concat(
            [X_filtered[X_filtered["Death"] == 1],
             self.old_saved_dead_individuals,
             self.saved_dead_individuals],
            axis=0
        )
        return X_filtered

    def balance_data(self, X_filtered):
        df_dead = self.saved_dead_individuals
        df_not_dead = X_filtered[X_filtered[self.response_parameter] == 0.0]
        df_not_dead = df_not_dead.sample(
            n=df_dead.shape[0],
            replace=False,
            random_state=self.seed,
        )
        df_balanced = pd.concat([df_dead, df_not_dead])
        return df_balanced

    def sensitivity_study(self, df, num_top_features):
        """
        Sensitivity study for the correlations between features
        and 'Death'.
        """
        real_base_corr = df.drop(self.symptoms, axis=1).corr()

        real_base_neg_corr = \
            real_base_corr[self.response_parameter].sort_values().head(30)
        real_base_pos_corr = \
            real_base_corr[self.response_parameter].sort_values().tail(30)

        top_pos = \
            real_base_pos_corr.index[(-1 - num_top_features):-1].to_list()
        top_neg = \
            real_base_neg_corr.index[:num_top_features].to_list()

        relevant_features = top_neg + top_pos

        for v in self.vaccines:
            if v not in relevant_features:
                relevant_features += [v]

        return relevant_features

    # Observe the features, treatments and outcomes of one or more individuals
    def observe(self, features, actions, outcomes):
        """Observe features, actions and outcomes.
        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)
        The function is used to adapt a model to the observed
        outcomes, given the actions and features. I suggest you create
        a model that estimates P(y | x,a) and fit it as appropriate.
        If the model cannot be updated incrementally, you can save all
        observed x,a,y triplets in a database and retrain whenever you
        obtain new data.
        Pseudocode:
            self.data.append(features, actions, outcomes)
            self.model.fit(data)
        """
        A = actions
        Y = outcomes

        # accumulating and storing database of vaccinated and dead people
        # and return filtered outcomes, i.e., the individuals that received
        # the vaccines and got covid.
        X_filtered = self.save_dead_individuals_for_observe(A, Y)

        # save baseline statistics in order to be compared with
        # future utilities:

        # For Vaccine 1
        # P(explanatory)
        _prob_explanatory = X_filtered.groupby(["Vaccine1"]).size().div(
            len(X_filtered))

        # P(response, explanatory)
        _prob_response_explanatory = X_filtered.groupby(
            ["Death", "Vaccine1"]).size().div(len(X_filtered))

        self.observed_P_D_given_v1 = _prob_response_explanatory.div(
            _prob_explanatory, axis=0)

        self.expected_num_D_given_v1 = \
            len(X_filtered) * self.observed_P_D_given_v1

        # For Vaccine 2
        # P(explanatory)
        _prob_explanatory = X_filtered.groupby(["Vaccine2"]).size().div(
            len(X_filtered))

        # P(response, explanatory)
        _prob_response_explanatory = X_filtered.groupby(
            ["Death", "Vaccine2"]).size().div(len(X_filtered))

        self.observed_P_D_given_v2 = _prob_response_explanatory.div(
            _prob_explanatory, axis=0)

        self.expected_num_D_given_v2 = \
            len(X_filtered) * self.observed_P_D_given_v2

        # For Vaccine 3
        # P(explanatory)
        _prob_explanatory = X_filtered.groupby(["Vaccine3"]).size().div(
            len(X_filtered))

        # P(response, explanatory)
        _prob_response_explanatory = X_filtered.groupby(
            ["Death", "Vaccine3"]).size().div(len(X_filtered))

        self.observed_P_D_given_v3 = X_filtered.groupby(
            ["Vaccine3"]).size().div(len(X_filtered))

        self.expected_num_D_given_v3 = \
            len(X_filtered) * self.observed_P_D_given_v3

        # For Vaccine 1 or 2 or 3
        self.observed_P_D_given_v123 = \
            sum(X_filtered["Death"]) / len(X_filtered)

        self.expected_num_D_given_v123 = \
            len(X_filtered) * self.observed_P_D_given_v123

        self.not_vaccinated = None
        self.vaccinated = None
        self.old_saved_dead_individuals = None

        """# balancing target labels in order to properly fit the ML model
        df_balanced = self.balance_data(X_filtered)

        # call a new pipeline to perform a randomized CV search
        self.create_new_ml_pipeline()

        # perform a sensitivity study (correlation study) between
        # features and 'Death' in order to pick up the most predictive
        # top 10 negative and top 10 positive features, plus actions.
        relevant_features = self.sensitivity_study(
            df=df_balanced,
            num_top_features=10
        )

        self.search_pipeline.fit(
            df_balanced[relevant_features],
            df_balanced["Death"]
        )
        print(f'Best Cross-Validated mean score: '
              f'{self.search_pipeline.best_score_}')"""

    def get_utility(self):
        """ Obtain the empirical utility of the policy on a set of one or
        more people.
        If there are t individuals with x features, and the action

        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)
        Returns:
        Empirical utility of the policy on this data.
        """
        # update the dead individuals base
        self.saved_dead_individuals = pd.concat(
            [self.old_saved_dead_individuals,
             self.saved_dead_individuals],
            axis=0
        )

        # balance target labels in order to properly fit the ML model
        df_balanced = self.balance_data(self.vaccinated)

        # call a new pipeline to perform a randomized CV search
        self.create_new_ml_pipeline()

        # perform a sensitivity study (correlation study) between
        # features and 'Death' in order to pick up the most predictive
        # top 10 negative and top 10 positive features, plus actions.
        relevant_features = self.sensitivity_study(
            df=df_balanced,
            num_top_features=10
        )

        # fit the pipeline
        self.search_pipeline.fit(
            df_balanced[relevant_features],
            df_balanced["Death"]
        )
        print(f'Best Cross-Validated mean score: '
              f'{self.search_pipeline.best_score_}')

        # pick the best fitted model
        model = self.search_pipeline.best_estimator_

        # predict probabilities for not vaccinated individuals
        actions_true = np.ones(shape=(self.not_vaccinated.shape[0], 1))
        actions_false = np.zeros(shape=(self.not_vaccinated.shape[0], 1))

        steps = [
            [actions_true, actions_false, actions_false],
            [actions_false, actions_true, actions_false],
            [actions_false, actions_false, actions_true],
        ]

        self.saved_pred = []
        self.saved_pob = []

        for s in steps:
            self.not_vaccinated['Vaccine1'] = s[0]
            self.not_vaccinated['Vaccine2'] = s[1]
            self.not_vaccinated['Vaccine3'] = s[2]

            self.saved_pred.append(
                model.predict(self.not_vaccinated[relevant_features]))

            self.saved_pob.append(
                model.predict_proba(self.not_vaccinated[relevant_features]))

        self.not_vaccinated['Vaccine3'] = actions_false

        A = np.zeros(shape=self.not_vaccinated.iloc[:, -3:].shape)
        A = pd.DataFrame(A, columns=self.vaccines)
        A.index = self.not_vaccinated.index

        self.count_num_deaths = 0
        for indx, indv in enumerate(self.not_vaccinated.index):
            decision = np.argmax(
                np.array([
                    self.saved_pob[0][indx][0],
                    self.saved_pob[1][indx][0],
                    self.saved_pob[2][indx][0]
                ])
            )
            self.count_num_deaths += self.saved_pred[decision][indx]

            if decision == 0:
                A["Vaccine1"][indv] = 1
            elif decision == 1:
                A["Vaccine2"][indv] = 1
            elif decision == 2:
                A["Vaccine3"][indv] = 1

        return A, self.count_num_deaths

    def get_actions(self, features):
        """Get actions for one or more people.
        Args:
        features (t*|X| array)
        Returns:
        actions (t*|A| array)
        Here you should take the action maximising expected utility
        according to your model. This model can be arbitrary, but
        should be adapted using the observe() method.
        Pseudocode:
           for action in appropriate_action_set:
                p = self.model.get_probabilities(features, action)
                u[action] = self.get_expected_utility(action, p)
           return argmax(u)
        You are expected to create whatever helper functions you need.
        """
        self.split_groups(features)

        # For 1st vaccination round: random decision
        # We vaccinate everybody, assuming the vaccines are already
        # approved by authorities, and secure for any type (age, gender)
        # of people.
        if self.vaccination_stage == 0:
            for indv in self.not_vaccinated.index:
                self.not_vaccinated[np.random.choice(self.vaccines)][indv] = 1

            A = self.not_vaccinated.iloc[:, -3:]
            self.vaccination_stage += 1
            return A

        if self.vaccination_stage > 0:
            # get actions based on our utility
            A, count_n_deaths = self.get_utility()

            if count_n_deaths <= 1.2 * self.expected_num_D_given_v123:
                self.not_vaccinated["Vaccine1"] = A["Vaccine1"]
                self.not_vaccinated["Vaccine2"] = A["Vaccine2"]
                self.not_vaccinated["Vaccine3"] = A["Vaccine3"]
                self.vaccination_stage += 1
                return self.not_vaccinated.iloc[:, -3:]

            else:
                for indv in self.not_vaccinated.index:
                    self.not_vaccinated[np.random.choice(self.vaccines)][indv] = 1

                A = self.not_vaccinated.iloc[:, -3:]
                self.vaccination_stage += 1
                return A

    def split_groups(self, X):
        # ~Vaccine1 and ~Vaccine2 and ~Vaccine3
        # goal: minimize the number of death between vaccinated people
        not_vaccinated = X[X['Vaccine1'] == 0]
        not_vaccinated = not_vaccinated[not_vaccinated['Vaccine2'] == 0]
        self.not_vaccinated = not_vaccinated[not_vaccinated['Vaccine3'] == 0]

        # Vaccine1 or Vaccine2 or Vaccine3
        # goal: use this data to get a policy under an utility
        vaccinated = X[X['Vaccine1'] == 1]
        vaccinated = pd.concat([vaccinated, X[X['Vaccine2'] == 1]], axis=0)
        self.vaccinated = \
            pd.concat([vaccinated, X[X['Vaccine3'] == 1]], axis=0)

        # dead individuals after vaccination
        saved_dead_individuals = vaccinated[vaccinated['Death'] == 1]
        self.old_saved_dead_individuals = saved_dead_individuals[
            saved_dead_individuals['Covid-Positive'] == 1]
