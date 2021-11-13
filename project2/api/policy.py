try:
    from models import DNN_CV
except:
    from project2.api.models import DNN_CV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Policy:
    """ A policy for treatment/vaccination. """

    @staticmethod
    def plot_fairness_age(df, title=''):
        title_size = 12
        l30 = df[df["Age"] < 30]
        bt3060 = df[
            (df["Age"] > 30) & (df["Age"] < 60)]
        g60 = df[df["Age"] > 60]

        df1 = pd.DataFrame({
            'Vaccines': [
                'Vaccine1', 'Vaccine1', 'Vaccine1',
                'Vaccine2', 'Vaccine2', 'Vaccine2',
                'Vaccine3', 'Vaccine3', 'Vaccine3'
            ],
            'Ages': [
                "l30", "bt3060", "g60",
                "l30", "bt3060", "g60",
                "l30", "bt3060", "g60",
            ],
            'Frequency': [
                l30.groupby("Vaccine1").size().iloc[1],
                bt3060.groupby("Vaccine1").size().iloc[1],
                g60.groupby("Vaccine1").size().iloc[1],
                l30.groupby("Vaccine2").size().iloc[1],
                bt3060.groupby("Vaccine2").size().iloc[1],
                g60.groupby("Vaccine2").size().iloc[1],
                l30.groupby("Vaccine3").size().iloc[1],
                bt3060.groupby("Vaccine3").size().iloc[1],
                g60.groupby("Vaccine3").size().iloc[1],
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Ages",
            data=df1,
            palette="Accent_r",
        )

        if title != '':
            plt.title(f"{title}", size=title_size)
        else:
            plt.title("Vaccinated Fairness among ages", size=title_size)
        plt.ylabel("Frequency", size=title_size)
        plt.xlabel("Vaccines", size=title_size)
        plt.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.show()

    @staticmethod
    def plot_fairness_gender(df, title=''):
        title_size = 12
        df1 = pd.DataFrame({
            'Vaccines': [
                'Vaccine1', 'Vaccine1',
                'Vaccine2', 'Vaccine2',
                'Vaccine3', 'Vaccine3'
            ],
            'Genders': [
                "female", "male",
                "female", "male",
                "female", "male"
            ],
            'Frequency': [
                df.groupby(["Gender", "Vaccine1"]).size().iloc[1],
                df.groupby(["Gender", "Vaccine1"]).size().iloc[3],
                df.groupby(["Gender", "Vaccine2"]).size().iloc[1],
                df.groupby(["Gender", "Vaccine2"]).size().iloc[3],
                df.groupby(["Gender", "Vaccine3"]).size().iloc[1],
                df.groupby(["Gender", "Vaccine3"]).size().iloc[3],
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Genders",
            data=df1,
            palette="Accent_r",
            # estimator=np.sum,
            # ax=ax1
        )

        if title != '':
            plt.title(f"{title}", size=title_size)
        else:
            plt.title("Vaccinated Fairness among genders", size=title_size)
        plt.ylabel("Frequency", size=title_size)
        plt.xlabel("Vaccines", size=title_size)
        plt.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.show()

    @staticmethod
    def plot_fairness_income(df, title=''):
        title_size = 12
        _0k2k = df[df["Income"] < 2000]
        _2k5k = df[(df["Income"] >= 2000) & (df["Income"] < 5000)]
        _5k10k = df[(df["Income"] >= 5000) & (df["Income"] < 10000)]
        _10k20k = df[(df["Income"] >= 10000) & (df["Income"] < 20000)]
        geq20k = df[df["Income"] >= 20000]

        df1 = pd.DataFrame({
            'Vaccines': [
                'Vaccine1', 'Vaccine1', 'Vaccine1', 'Vaccine1', 'Vaccine1',
                'Vaccine2', 'Vaccine2', 'Vaccine2', 'Vaccine2', 'Vaccine2',
                'Vaccine3', 'Vaccine3', 'Vaccine3', 'Vaccine3', 'Vaccine3'
            ],
            'Incomes': [
                "0k2k", "2k5k", "5k10k", "10k20k", "20k30k",
                "0k2k", "2k5k", "5k10k", "10k20k", "20k30k",
                "0k2k", "2k5k", "5k10k", "10k20k", "20k30k",
            ],
            'Frequency': [
                _0k2k.groupby("Vaccine1").size().iloc[1],
                _2k5k.groupby("Vaccine1").size().iloc[1],
                _5k10k.groupby("Vaccine1").size().iloc[1],
                _10k20k.groupby("Vaccine1").size().iloc[1],
                geq20k.groupby("Vaccine1").size().iloc[1],
                _0k2k.groupby("Vaccine2").size().iloc[1],
                _2k5k.groupby("Vaccine2").size().iloc[1],
                _5k10k.groupby("Vaccine2").size().iloc[1],
                _10k20k.groupby("Vaccine2").size().iloc[1],
                geq20k.groupby("Vaccine2").size().iloc[1],
                _0k2k.groupby("Vaccine3").size().iloc[1],
                _2k5k.groupby("Vaccine3").size().iloc[1],
                _5k10k.groupby("Vaccine3").size().iloc[1],
                _10k20k.groupby("Vaccine3").size().iloc[1],
                geq20k.groupby("Vaccine3").size().iloc[1],
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Incomes",
            data=df1,
            palette="Accent_r",
            # estimator=np.sum,
            # ax=ax1
        )

        if title != '':
            plt.title(f"{title}", size=title_size)
        else:
            plt.title("Fairness among incomes", size=title_size)
        plt.ylabel("Frequency", size=title_size)
        plt.xlabel("Vaccines", size=title_size)
        plt.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5)
        plt.show()

    def __init__(self, n_actions, action_set, plot_fairness=False,
                 seed=1, n_jobs=-1):
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
        self.search_pipeline = None

        self.seed = seed
        self.n_jobs = n_jobs

        # from get_action step
        self.not_vaccinated = None
        self.saved_vaccinated = pd.DataFrame()
        self.saved_dead_individuals = pd.DataFrame()

        # random expected utility estimations
        self.random_expected_utilities = []

        # from observe step
        self.observed_utilities = []
        self.observed_expected_utilities = []
        self.observed_expected_utility = None

        # from get_utility step
        self.ML_expected_utilities = []
        self.ML_expected_utility = None

        # step checker
        self.was_observed = 0

        # list containing the policy decision
        self.policy_decisions = []
        self.last_policy = None

        # Fairness
        self.plot_fairness = plot_fairness

        # relevant features
        self.relevant_features = None

    def create_new_ml_pipeline(self):
        """Create a pipeline with a randomized search CV and random
        forest classifier."""
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

    def save_database_and_filter_outcomes(self, A, Y):
        """Store dead individual who will be object of our study, because
        we want to understand why they come to die after have taken a vaccine
        and been infected by the covid-19. Also, filter outcomes to
        only return vaccinated individuals who got Covid-19 after
        vaccination."""
        idx = A.index.to_list()
        Y.index = idx
        X_ = self.not_vaccinated[self.not_vaccinated.index.isin(idx)]
        X_ = X_.drop(self.symptoms + ["Death"] + self.vaccines,
                     axis=1)  # dropping old features

        # concatenating applied actions and observed outcomes.
        filtered_outcomes = pd.concat([Y, X_, A], axis=1)

        # saving the ones who survived.
        self.saved_vaccinated = pd.concat(
            [filtered_outcomes,
             self.saved_vaccinated],
            axis=0
        )

        # resetting indexes
        self.saved_vaccinated.index = \
            [i for i in range(len(self.saved_vaccinated))]

        # filtering only covid-positive ones.
        filtered_outcomes = \
            filtered_outcomes[filtered_outcomes["Covid-Positive"] == 1]

        # saving the ones who came to die.
        self.saved_dead_individuals = pd.concat(
            [filtered_outcomes[filtered_outcomes["Death"] == 1],
             self.saved_dead_individuals],
            axis=0
        )

        # resetting indexes
        self.saved_dead_individuals.index = \
            [i for i in range(len(self.saved_dead_individuals))]

        return filtered_outcomes

    def balance_data(self):
        """Balance target labels in order to properly train a
        machine learning model."""

        df_dead = self.saved_dead_individuals

        df_dead = resample(
            df_dead,
            replace=True,
            n_samples=(3 * len(df_dead)),
            random_state=self.seed,
            stratify=None
        )

        df_not_dead = self.saved_vaccinated[
            self.saved_vaccinated[self.response_parameter] == 0.0]

        df_not_dead = df_not_dead[df_not_dead["Covid-Positive"] == 1]

        df_not_dead = df_not_dead.sample(
            n=df_dead.shape[0],
            replace=True,
            random_state=self.seed,
        )

        self.df_balanced = pd.concat([df_dead, df_not_dead])
        return self.df_balanced

    def sensitivity_study(self, df, num_top_features):
        """Sensitivity study for the correlations between features
        and 'Death'."""
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

        self.relevant_features = relevant_features
        return relevant_features

    # Observe the features, treatments and outcomes of one or more individuals
    def observe(self, actions, outcomes):
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

        if self.plot_fairness:
            self.plot_fairness_age(
                df=self.not_vaccinated,
                title=f"Fairness among ages in actions "
                      f"(stage: {self.vaccination_stage}, "
                      f"policy: {self.last_policy})"
            )

            self.plot_fairness_gender(
                df=self.not_vaccinated,
                title=f"Fairness between gender in actions "
                      f"(stage: {self.vaccination_stage}, "
                      f"policy: {self.last_policy})"
            )

            self.plot_fairness_income(
                df=self.not_vaccinated,
                title=f"Fairness among incomes in actions "
                      f"(stage: {self.vaccination_stage}, "
                      f"policy: {self.last_policy})"
            )

        # accumulating and storing database of vaccinated and dead people
        # and return filtered outcomes, i.e., the individuals that received
        # the vaccines and got covid.
        filtered_outcomes = self.save_database_and_filter_outcomes(A, Y)

        # save baseline statistics in order to be compared with
        # future utilities:

        # P_hat(D|V1 or V2 or V3):
        observed_P_D_given_v123 = \
            sum(filtered_outcomes["Death"]) / len(self.not_vaccinated)

        self.observed_utilities.append(observed_P_D_given_v123)

        # E(Num of deaths | V1 or V2 or V3):
        self.observed_expected_utility = \
            len(self.not_vaccinated) * observed_P_D_given_v123

        self.observed_expected_utilities.append(
            self.observed_expected_utility)

        # step check
        self.was_observed = 1

        # saving baseline for the first vaccination round
        if self.last_policy == "Random":
            self.random_expected_utilities.append(
                self.observed_expected_utility)

        print(f"Observed Expected Utility: "
              f"{self.observed_expected_utility}")

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
        # balance target labels in order to properly fit the ML model.
        df_balanced = self.balance_data()

        # call a new pipeline to perform a randomized CV search.
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

        # pick the vaccine with higher probability of survival,
        # and count the number of predicted death E(D|V1orV2orV3).
        self.ML_expected_utility = 0
        for indx, indv in enumerate(self.not_vaccinated.index):
            decision = np.argmax(
                np.array([
                    self.saved_pob[0][indx][0],
                    self.saved_pob[1][indx][0],
                    self.saved_pob[2][indx][0]
                ])
            )
            self.ML_expected_utility += self.saved_pred[decision][indx]

            if decision == 0:
                A["Vaccine1"][indv] = 1
            elif decision == 1:
                A["Vaccine2"][indv] = 1
            elif decision == 2:
                A["Vaccine3"][indv] = 1

        self.ML_expected_utilities.append(self.ML_expected_utility)
        print(f"ML Expected Utility: {self.ML_expected_utility}")
        return A, self.ML_expected_utility

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
        # step check
        if (self.was_observed == 1) or (self.vaccination_stage == 0):
            self.was_observed = 0

        else:
            raise ValueError("ERROR: Observe step was not applied before "
                             "a new get_action.")

        # vide docstrings in this function.
        self.split_groups(features)

        # For 1st vaccination round: random decisions.
        # We vaccinate everybody, assuming that the vaccines are already
        # approved by authorities, and secure for any type (age, gender)
        # of people.
        if self.vaccination_stage == 0:
            for indv in self.not_vaccinated.index:
                self.not_vaccinated[np.random.choice(self.vaccines)][indv] = 1

            A = self.not_vaccinated.iloc[:, -3:]
            self.vaccination_stage += 1
            self.last_policy = "Random"
            self.policy_decisions.append(self.last_policy)
            return A

        if self.vaccination_stage > 0:
            # get actions based on our utility
            A, ML_expected_utility = self.get_utility()

            # security threshold to use our ML's action recommendations.
            if ML_expected_utility <= \
                    np.mean(self.random_expected_utilities):

                self.not_vaccinated["Vaccine1"] = A["Vaccine1"]
                self.not_vaccinated["Vaccine2"] = A["Vaccine2"]
                self.not_vaccinated["Vaccine3"] = A["Vaccine3"]
                self.vaccination_stage += 1
                self.last_policy = "ML"
                self.policy_decisions.append(self.last_policy)
                return self.not_vaccinated.iloc[:, -3:]

            # if our ML model does not hit our security threshold,
            # then apply random decisions.
            else:
                for indv in self.not_vaccinated.index:
                    self.not_vaccinated[np.random.choice(self.vaccines)][
                        indv] = 1

                A = self.not_vaccinated.iloc[:, -3:]
                self.vaccination_stage += 1
                self.last_policy = "Random"
                self.policy_decisions.append(self.last_policy)
                return A

    def split_groups(self, X):
        """From a raw sample space, we split the data in 3 groups.
        1st. the not vaccinated individuals that will be object to
        our recommendations. 2nd. the vaccinated people that will be
        object to study the influences of the vaccines. 3rd. the dead
        individuals who we will have in order to adapt and fit a
        machine learning, and understand why those individuals come
        to die after vaccination."""
        # ~Vaccine1 and ~Vaccine2 and ~Vaccine3
        # goal: minimize the number of death between vaccinated people
        not_vaccinated = X[X['Vaccine1'] == 0]
        not_vaccinated = not_vaccinated[not_vaccinated['Vaccine2'] == 0]
        self.not_vaccinated = not_vaccinated[not_vaccinated['Vaccine3'] == 0]

        # Vaccine1 or Vaccine2 or Vaccine3
        # goal: use this data to get a policy under an utility
        vaccinated = X[X['Vaccine1'] == 1]
        vaccinated = pd.concat([vaccinated, X[X['Vaccine2'] == 1]], axis=0)
        vaccinated = pd.concat([vaccinated, X[X['Vaccine3'] == 1]], axis=0)
        vaccinated = vaccinated[vaccinated["Covid-Positive"] == 1]

        if self.plot_fairness:
            self.plot_fairness_age(
                df=vaccinated,
                title=f"Fairness among ages from sample data "
                      f"(stage: {self.vaccination_stage})"
            )
            self.plot_fairness_gender(
                df=vaccinated,
                title=f"Fairness between genders from sample data "
                      f"(stage: {self.vaccination_stage})"
            )
            self.plot_fairness_income(
                df=vaccinated,
                title=f"Fairness among incomes from sample data "
                      f"(stage: {self.vaccination_stage})"
            )

        self.saved_vaccinated = pd.concat(
            [vaccinated,
             self.saved_vaccinated],
            axis=0
        )

        # resetting indexes
        self.saved_vaccinated.index = \
            [i for i in range(len(self.saved_vaccinated))]

        # dead individuals after vaccination
        saved_dead_individuals = vaccinated[vaccinated['Death'] == 1]
        saved_dead_individuals = saved_dead_individuals[
            saved_dead_individuals['Covid-Positive'] == 1]

        self.saved_dead_individuals = pd.concat(
            [saved_dead_individuals,
             self.saved_dead_individuals],
            axis=0
        )

        # resetting indexes
        self.saved_dead_individuals.index = \
            [i for i in range(len(self.saved_dead_individuals))]
