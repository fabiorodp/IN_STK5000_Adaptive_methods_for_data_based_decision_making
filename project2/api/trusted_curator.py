try:
    from users import credentials
except:
    from project2.api.users import credentials
    import project2.src.covid.simulator as simulator

import pandas as pd


class TrustedCurator:

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
        self.mode = 'on'
        self.user = ''
        if user in credentials.keys():
            if password == credentials[user]:
                self.user = user
                if user == 'master':
                    self.mode = mode
            else:
                raise ValueError("ERROR: Wrong user or password.")
        else:
            raise ValueError("ERROR: Wrong user or password.")

        self.X = None

    def get_features(self):
        """
        :return: X
        """
        if self.mode == 'off':
            population = simulator.Population(
                n_genes=128,
                n_vaccines=3,
                n_treatments=4
            )

            self.X = population.generate(
                n_individuals=10000
            )

            self.X = pd.DataFrame(self.X)

            self.X.columns = self.labels(
                num_genes=128,
                num_vaccines=3
            )

            return self.X

        else:
            raise NotImplemented

    def get_outcomes(self, features, action):
        """
        Give:
        X, A

        :return: Y = X x A x Y
        """
        pass
