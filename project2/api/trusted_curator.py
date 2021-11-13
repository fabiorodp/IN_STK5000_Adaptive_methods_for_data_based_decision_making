try:
    from users import credentials
except:
    from project2.api.users import credentials
    import project2.api.simulator as simulator

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
        :param x: df['Death']
        :param R: df['Death'].unique()
        :param u: self.u(x, values)
        :param sensitivity:
        :param epsilon: 0.1, 0.2, ..., 0.9
        :param n: len(df["Death"])
        :return:
        """
        # utility score for each element in R
        scores = u(x, R)

        # applying mechanism
        probs = np.exp(epsilon * scores / sensitivity)

        # normalizing
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

    def __init__(self, user, password, mode='on', epsilon=0.1):
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

        self.epsilon = epsilon

    def __get_dataX(self, n_population):
        """Private class function to query real data."""
        X = self.population.generate(
            n_individuals=n_population
        )

        X = pd.DataFrame(X)

        X.columns = self.labels(
            num_genes=128,
            num_vaccines=3
        )

        if self.mode == 'off':
            return X

        else:
            X["Age"].apply(lambda x: int(x))  # date of birth is protected

            for i in range(len(X)):  # gender is protected
                if np.random.randint(0, 2) == 0:
                    X["Gender"][i] = X["Gender"][i]
                else:
                    X["Gender"][i] = np.random.randint(0, 2)

            for i in range(len(X)):  # income is protected
                sensitivity, epsilon = 1, 0.1
                X['Income'][i] += np.random.laplace(0, sensitivity / epsilon)

            return X

    def __get_dataY(self, individuals_idxs, actions):
        """Private class function to query real data."""
        Y = self.population.vaccinate(
            individuals_idxs,
            actions.values
        )

        Y = pd.DataFrame(Y)

        Y.columns = ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell',
                     'Fever', 'Headache', 'Pneumonia', 'Stomach',
                     'Myocarditis', 'Blood-Clots', 'Death']

        if self.mode == 'off':
            return Y

        else:
            new_Y = self.exponential(
                x=Y["Death"],
                R=Y["Death"].unique(),
                u=self.u,
                sensitivity=1,
                epsilon=self.epsilon,
                n=len(Y)
            )

            Y["Death"] = new_Y
            return Y

    def get_features(self, n_population=100000):
        """
        :return: X
        """
        self.X = self.__get_dataX(
            n_population=n_population,
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
        )
        return self.Y
