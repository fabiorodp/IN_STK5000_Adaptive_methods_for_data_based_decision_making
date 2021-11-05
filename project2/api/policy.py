try:
    from models import DNN_CV
except:
    from project2.api.models import DNN_CV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class Policy:
    """ A policy for treatment/vaccination. """

    @staticmethod
    def balance_data(X, param):
        df_dead = X[X[param] == 1.0]
        df_not_dead = X[X[param] == 0.0].iloc[:df_dead.shape[0], :]
        df_balanced = pd.concat([df_dead, df_not_dead])
        return df_balanced

    @staticmethod
    def sensitivity_study(X, num_top_features=10):
        """Sensitivity study for the correlations between features
        and 'Death'."""
        parameter = "Death"
        real_base = X[X['Covid-Positive'] == 1].drop(
            ['Covid-Positive'], axis=1)

        real_base = real_base.drop(
            ['No_Taste/Smell', 'Fever', 'Headache', 'Pneumonia', 'Stomach',
             'Myocarditis', 'Blood-Clots'], axis=1
        )

        real_base_corr = real_base.corr()
        real_base_neg_corr = real_base_corr[parameter].sort_values().head(30)
        real_base_pos_corr = real_base_corr[parameter].sort_values().tail(30)

        top_pos = real_base_pos_corr.index[(-1-num_top_features):-1].to_list()
        top_neg = real_base_neg_corr.index[:num_top_features].to_list()

        return top_neg, top_pos, real_base

    def __init__(self, n_actions, action_set, seed=1, n_jobs=-1):
        """ Initialise.
        Args:
        n_actions (int): the number of actions
        action_set (list): the set of actions
        """
        np.random.seed(1)
        self.n_actions = n_actions
        self.action_set = action_set
        print("Initialising policy with ", n_actions, "actions")
        print("A = {", action_set, "}")

        self.stage = 0
        self.model = None
        self.search = None
        self.relevant_features = None

        self.seed = seed
        self.n_jobs = n_jobs

    def create_model(self, X, n_top_features=10):
        # Get top 10 most positive and negative correlated to 'Death'
        top_neg, top_pos, real_base = self.sensitivity_study(
            X=X,
            num_top_features=n_top_features
        )

        # Balance targets' label
        df_balanced = self.balance_data(
            X=real_base,
            param='Death'
        )

        # Check if actions are included.
        # If not, then include it to relevant features.
        rf = top_neg + top_pos
        for e in ['Vaccine1', 'Vaccine2', 'Vaccine3']:
            if e not in rf:
                rf += [e]

        self.relevant_features = rf

        XandA = df_balanced[self.relevant_features]
        Y = df_balanced['Death']

        steps = [
            ('lr',
             LogisticRegression(
                 solver='saga',
                 random_state=self.seed,
                 n_jobs=self.n_jobs,
                 verbose=False
             ))
        ]

        pipeline = Pipeline(steps)

        param_dist = {
            'lr__penalty': ['l2', 'l1'],
            'lr__max_iter': [5, 9, 10, 15, 20, 25, 50],
            'lr__C': [0.1, 0.25, 0.5, 0.75, 1]
        }

        self.search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=25,
            scoring='accuracy',
            refit=True,
            cv=5,
            random_state=self.seed
        )
        self.search.fit(XandA, Y)
        print(f'Best Cross-Validated mean score: {self.search.best_score_}')

        self.model = self.search.best_estimator_

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
        self.Xprime = pd.concat(
            [features.drop(['Death', 'Vaccine1', 'Vaccine2', 'Vaccine3'], axis=1),
             actions,
             outcomes['Death']
             ],
            axis=1
        )

        self.XandAandY = self.Xprime[self.Xprime['Covid-Positive'] == 1].drop(['Covid-Positive'], axis=1)

        # Balance
        df_dead = self.XandAandY[self.XandAandY['Death'] == 1.0]
        df_not_dead = self.XandAandY[self.XandAandY['Death'] == 0.0].iloc[:df_dead.shape[0], :]
        self.df_balanced = pd.concat([df_dead, df_not_dead])

        self.search.fit(self.df_balanced[self.relevant_features], self.df_balanced['Death'])
        self.model = self.search.best_estimator_
        print(f'Best Cross-Validated mean score: {self.search.best_score_}')

        return

    def get_utility(self, features, actions, outcomes):
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
        utility = np.sum(outcomes == 0.) / features.shape[0]
        print(f"Average survival {utility}")
        return utility

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
        X = features[self.relevant_features].drop(
            ['Vaccine1', 'Vaccine2', 'Vaccine3'], axis=1)

        actions_true = np.ones(shape=(X.shape[0], 1))
        actions_false = np.zeros(shape=(X.shape[0], 1))

        self.steps = [
            [actions_false, actions_false, actions_false],
            [actions_true, actions_false, actions_false],
            [actions_false, actions_true, actions_false],
            [actions_false, actions_false, actions_true],
        ]

        self.avg_sur = []
        for a in self.steps:

            self.X_ = pd.concat(
                [X,
                 pd.DataFrame(a[0], columns=['Vaccine1']),
                 pd.DataFrame(a[1], columns=['Vaccine2']),
                 pd.DataFrame(a[2], columns=['Vaccine3'])
                 ],
                axis=1
            )

            # P_a1(Y | X, A), P_a2(Y | X, A), P_a3(Y | X, A), P_a4(Y | X, A)
            self.pred = self.model.predict(self.X_)  # 0 or 1
            self.prob = self.model.predict_proba(self.X_)  # P_a1(Y | X, A)

            u = self.get_utility(
                features=self.X_,
                actions=a,
                outcomes=self.pred
            )

            self.avg_sur.append(u)

        # MAXIMIZE EXPECTED UTILITY
        self.u_max = np.argmax(np.array(self.avg_sur))

        self.A = pd.DataFrame(
            np.concatenate(
                (self.steps[self.u_max][0], self.steps[self.u_max][1],
                 self.steps[self.u_max][2]),
                axis=1),
            columns=['Vaccine1', 'Vaccine2', 'Vaccine3']
        )

        return self.A
