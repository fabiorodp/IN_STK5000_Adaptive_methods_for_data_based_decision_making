try:
    from users import credentials
except:
    from project2.api.users import credentials
    import project2.src.covid.simulator as simulator

import pandas as pd
import numpy as np


class TrustedCurator:

    @staticmethod
    def u(x, value):
        v_cnts = x.value_counts()
        return v_cnts[value] / v_cnts.sum()

    @staticmethod
    def exponential(x, R, u, sensitivity, epsilon, n=1):
        """
        :param x: df['Vaccine1']
        :param R: df['Vaccine1'].unique()
        :param u:
        :param sensitivity:
        :param epsilon:
        :param n:
        :return:
        """
        scores = u(x, R)  # score each element in R
        probs = np.exp(epsilon * scores / 2 / sensitivity)
        probs /= probs.sum()
        return np.random.choice(R, n, p=probs)

    @staticmethod
    def labels(num_genes=128, num_vaccines=3):
        return ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell',
                'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis',
                'Blood-Clots', 'Death', 'Age', 'Gender', 'Income'] + \
                [f'g{i}' for i in range(1, num_genes+1)] + \
                ['Asthma', 'Obesity', 'Smoking', 'Diabetes',
                 'Heart-disease', 'Hypertension'] +\
               [f'Vaccine{i}' for i in range(1, num_vaccines+1)]

    def __init__(self, user, password, mode='on'):
        # np.random.seed(1)
        self.mode = 'on'
        self.user = ''
        if (user in credentials.keys()) and (password == credentials[user]):
            self.user = user
            if user == 'master':
                self.mode = mode
        else:
            raise ValueError("ERROR: Wrong user or password.")

        self.population = simulator.Population(
            n_genes=128,
            n_vaccines=3,
            n_treatments=4
        )

    def __get_dataX(self, n_population, mode='on'):
        """Private class function to query real data."""
        X = self.population.generate(
            n_individuals=n_population
        )

        X = pd.DataFrame(X)

        X.columns = self.labels(
            num_genes=128,
            num_vaccines=3
        )
        return X

    def __get_dataY(self, individuals_idxs, actions, mode='on'):
        """Private class function to query real data."""
        Y = self.population.vaccinate(
            individuals_idxs,
            actions.values
        )

        Y = pd.DataFrame(Y)

        Y.columns = ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell',
                     'Fever', 'Headache', 'Pneumonia', 'Stomach',
                     'Myocarditis', 'Blood-Clots', 'Death']
        return Y

    def get_features(self, n_population=100000):
        """
        :return: X
        """
        self.X = self.__get_dataX(
            n_population=n_population,
            mode=self.mode
        )
        return self.X

    def get_outcomes(self, individuals_idxs, actions):
        """
        Give:
        X, A

        :return: Y = X x A x Y
        """
        self.Y = self.__get_dataY(
            individuals_idxs=individuals_idxs,
            actions=actions,
            mode=self.mode
        )
        return self.Y