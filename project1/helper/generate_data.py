from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class Space:
    """Class function to create the Synthetic data."""

    @staticmethod
    def init_random_space(N, add_treatment):
        """Assign independent distributed values among features."""
        _feature_covid_recovered = \
            np.random.binomial(1, 0.03, N)[:, np.newaxis]
        _feature_covid_positive = \
            np.random.binomial(1, 0.3, N)[:, np.newaxis]

        _feature_symptoms = np.random.randint(
            low=0,
            high=8,
            size=N
        )[:, np.newaxis]

        _feature_symptoms = \
            OneHotEncoder(sparse=False).fit_transform(_feature_symptoms)[:, 1:]

        _feature_death = np.random.binomial(1, 0.1, N)[:, np.newaxis]
        _feature_ages = np.random.randint(1, 100, N)[:, np.newaxis]
        _feature_gender = np.random.binomial(1, 0.5, N)[:, np.newaxis]
        _feature_income = np.random.normal(25000, 10000, N)[:, np.newaxis]
        _feature_income[_feature_income <= 10] = 0

        _feature_genes = {}
        for i in range(128):
            _feature_genes[f'g{i + 1}'] = np.random.binomial(1, 0.25, N)

        _feature_asthma = np.random.binomial(1, 0.07, N)[:, np.newaxis]
        _feature_obesity = np.random.binomial(1, 0.13, N)[:, np.newaxis]
        _feature_smoking = np.random.binomial(1, 0.19, N)[:, np.newaxis]
        _feature_diabetes = np.random.binomial(1, 0.10, N)[:, np.newaxis]
        _feature_heart_disease = \
            np.random.binomial(1, 0.10, N)[:, np.newaxis]
        _feature_hypertension = \
            np.random.binomial(1, 0.17, N)[:, np.newaxis]

        _feature_vaccines = np.random.randint(
            low=0,
            high=4,
            size=N
        )[:, np.newaxis]

        _feature_vaccines = \
            OneHotEncoder(sparse=False).fit_transform(_feature_vaccines)[:, 1:]

        _feature_treatment1 = np.zeros(N)[:, np.newaxis]
        _feature_treatment2 = np.zeros(N)[:, np.newaxis]
        for idx, e in enumerate(_feature_covid_positive):
            if e == 1:
                _feature_treatment1[idx] = np.random.binomial(1, 0.70)
                _feature_treatment2[idx] = np.random.binomial(1, 0.50)

        features = [
            pd.DataFrame(_feature_covid_recovered,
                         columns=['Covid-Recovered']),
            pd.DataFrame(_feature_covid_positive, columns=['Covid-Positive']),
            pd.DataFrame(_feature_symptoms, columns=['No_Taste/Smell', 'Fever',
                                                     'Headache', 'Pneumonia',
                                                     'Stomach', 'Myocarditis',
                                                     'Blood-Clots', ]),
            pd.DataFrame(_feature_death, columns=['Death']),
            pd.DataFrame(_feature_ages, columns=['Age']),
            pd.DataFrame(_feature_gender, columns=['Gender']),
            pd.DataFrame(_feature_income, columns=['Income']),
            pd.DataFrame(_feature_genes),
            pd.DataFrame(_feature_asthma, columns=['Asthma']),
            pd.DataFrame(_feature_obesity, columns=['Obesity']),
            pd.DataFrame(_feature_smoking, columns=['Smoking']),
            pd.DataFrame(_feature_diabetes, columns=['Diabetes']),
            pd.DataFrame(_feature_heart_disease, columns=['Heart-disease']),
            pd.DataFrame(_feature_hypertension, columns=['Hypertension']),
            pd.DataFrame(_feature_vaccines, columns=['Vaccine1', 'Vaccine2',
                                                     'Vaccine3']),
            pd.DataFrame(_feature_treatment1, columns=['Treatment1']),
            pd.DataFrame(_feature_treatment2, columns=['Treatment2']),
        ]

        if not add_treatment:
            df = pd.concat(features[:-2], axis=1)
        else:
            df = pd.concat(features, axis=1)

        # fixing income to zero for people below 18 years old
        df['Income'].where(df['Age'] > 18, 0, inplace=True)
        return df

    @staticmethod
    def defined_cond_probs_with_death(is_treatment_included=False):
        """Pre-define conditional probabilities for death."""
        ages_ = np.array([20, 40, 60, 80, 100])
        diabetes_ = np.array([0, 1])
        hypertension_ = np.array([0, 1])
        g1g2_ = np.array([0, 2])
        v1_ = np.array([0, 1])
        v2_ = np.array([0, 1])
        v3_ = np.array([0, 1])
        t1_ = np.array([0, 1])
        t2_ = np.array([0, 1])

        if not is_treatment_included:
            cond_probs = np.array(
                np.meshgrid(ages_, diabetes_, hypertension_, g1g2_, v1_, v2_, v3_)
            ).T.reshape(-1, 7)

            prob_ages_ = {20: 0.05, 40: 0.15, 60: 0.25, 80: 0.50, 100: 0.70}
            prob_diabetes_ = {0: 0., 1: 0.30}
            prob_hypertension_ = {0: 0., 1: 0.30}
            prob_g1g2_ = {0: 0., 2: 0.90}
            prob_v1_ = {0: 0., 1: -0.30}
            prob_v2_ = {0: 0., 1: -0.40}
            prob_v3_ = {0: 0., 1: -0.50}

            probs = np.zeros((len(cond_probs),))
            idx_probs = 0
            for idx_r, r in enumerate(cond_probs):

                p = prob_ages_[cond_probs[idx_r, 0]] + \
                    prob_diabetes_[cond_probs[idx_r, 1]] + \
                    prob_hypertension_[cond_probs[idx_r, 2]] + \
                    prob_g1g2_[cond_probs[idx_r, 3]] + \
                    prob_v1_[cond_probs[idx_r, 4]] + \
                    prob_v2_[cond_probs[idx_r, 5]] + \
                    prob_v3_[cond_probs[idx_r, 6]]

                if p <= 0.01:
                    probs[idx_probs] = 0.01
                elif p >= 0.9:
                    probs[idx_probs] = 0.9
                else:
                    probs[idx_probs] = p
                idx_probs += 1

            cond_probs = pd.concat(
                [
                    pd.DataFrame(
                        cond_probs,
                        columns=['Age', 'Diabetes', 'Hypertension',
                                 'G1+G2', 'V1', 'V2', 'V3']
                    ),
                    pd.DataFrame(
                        probs[:, np.newaxis],
                        columns=['Probabilities']
                    )
                ], axis=1)

            cond_probs_dict = {}
            for i in range(len(cond_probs)):
                key = f"{int(cond_probs.iloc[i, :]['Age'])}" + \
                      f"{int(cond_probs.iloc[i, :]['Diabetes'])}" + \
                      f"{int(cond_probs.iloc[i, :]['Hypertension'])}" + \
                      f"{int(cond_probs.iloc[i, :]['G1+G2'])}" + \
                      f"{int(cond_probs.iloc[i, :]['V1'])}" + \
                      f"{int(cond_probs.iloc[i, :]['V2'])}" + \
                      f"{int(cond_probs.iloc[i, :]['V3'])}"

                cond_probs_dict[key] = cond_probs.iloc[i, -1]

            return cond_probs, cond_probs_dict

        else:
            cond_probs = np.array(
                np.meshgrid(ages_, diabetes_, hypertension_, g1g2_, t1_, t2_)
            ).T.reshape(-1, 6)

            prob_ages_ = {20: 0.05, 40: 0.15, 60: 0.25, 80: 0.50, 100: 0.70}
            prob_diabetes_ = {0: 0., 1: 0.30}
            prob_hypertension_ = {0: 0., 1: 0.30}
            prob_g1g2_ = {0: 0., 2: 0.90}
            prob_t1_ = {0: 0., 1: -2.00}
            prob_t2_ = {0: 0., 1: -1.50}

            probs = np.zeros((len(cond_probs),))
            idx_probs = 0
            for idx_r, r in enumerate(cond_probs):

                p = prob_ages_[cond_probs[idx_r, 0]] + \
                    prob_diabetes_[cond_probs[idx_r, 1]] + \
                    prob_hypertension_[cond_probs[idx_r, 2]] + \
                    prob_g1g2_[cond_probs[idx_r, 3]] + \
                    prob_t1_[cond_probs[idx_r, 4]] + \
                    prob_t2_[cond_probs[idx_r, 5]]

                if p <= 0.01:
                    probs[idx_probs] = 0.01
                elif p >= 0.9:
                    probs[idx_probs] = 0.9
                else:
                    probs[idx_probs] = p
                idx_probs += 1

            cond_probs = pd.concat(
                [
                    pd.DataFrame(
                        cond_probs,
                        columns=['Age', 'Diabetes', 'Hypertension',
                                 'G1+G2', 'T1', 'T2']
                    ),
                    pd.DataFrame(
                        probs[:, np.newaxis],
                        columns=['Probabilities']
                    )
                ], axis=1)

            cond_probs_dict = {}
            for i in range(len(cond_probs)):
                key = f"{int(cond_probs.iloc[i, :]['Age'])}" + \
                      f"{int(cond_probs.iloc[i, :]['Diabetes'])}" + \
                      f"{int(cond_probs.iloc[i, :]['Hypertension'])}" + \
                      f"{int(cond_probs.iloc[i, :]['G1+G2'])}" + \
                      f"{int(cond_probs.iloc[i, :]['T1'])}" + \
                      f"{int(cond_probs.iloc[i, :]['T2'])}"

                cond_probs_dict[key] = cond_probs.iloc[i, -1]

            return cond_probs, cond_probs_dict

    @staticmethod
    def help_age(given_age):
        if given_age <= 20:
            return 20
        elif given_age <= 40:
            return 40
        elif given_age <= 60:
            return 60
        elif given_age <= 80:
            return 80
        else:
            return 100

    @staticmethod
    def help_gene(given_gene):
        if given_gene == 1:
            return 0
        else:
            return given_gene

    def __init__(self, N=100000, add_treatment=False, seed=1):
        np.random.seed(seed)
        self.N = N
        self.add_treatment = add_treatment
        self.space = self.init_random_space(self.N, self.add_treatment)
        self.cond_probs_with_death, self.cond_probs_with_death_dict = \
            self.defined_cond_probs_with_death(
                is_treatment_included=self.add_treatment)
        self.space_corr_mtx = self.space.corr()

    def assign_corr_death(self):
        if not self.add_treatment:
            cond_probs_dict = self.cond_probs_with_death_dict
            new_death = []
            for i in range(len(self.space)):
                if self.space.iloc[i]['Covid-Positive'] == 1:
                    find = f"{int(self.help_age(given_age=self.space.iloc[i]['Age']))}" + \
                           f"{int(self.space.iloc[i]['Diabetes'])}" + \
                           f"{int(self.space.iloc[i]['Hypertension'])}" + \
                           f"{int(self.help_gene(given_gene=self.space.iloc[i]['g1']+ self.space.iloc[i]['g2']))}" + \
                           f"{int(self.space.iloc[i]['Vaccine1'])}" + \
                           f"{int(self.space.iloc[i]['Vaccine2'])}" + \
                           f"{int(self.space.iloc[i]['Vaccine3'])}"
                    new_death.append(np.random.binomial(1, cond_probs_dict[find]))

                else:
                    new_death.append(np.random.binomial(1, 0.05))

            self.space['Death'] = new_death
            self.space_corr_mtx = self.space.corr()

        else:
            cond_probs_dict = self.cond_probs_with_death_dict
            new_death = []
            for i in range(len(self.space)):
                if self.space.iloc[i]['Covid-Positive'] == 1:
                    find = f"{int(self.help_age(given_age=self.space.iloc[i]['Age']))}" + \
                           f"{int(self.space.iloc[i]['Diabetes'])}" + \
                           f"{int(self.space.iloc[i]['Hypertension'])}" + \
                           f"{int(self.help_gene(given_gene=self.space.iloc[i]['g1'] + self.space.iloc[i]['g2']))}" + \
                           f"{int(self.space.iloc[i]['Treatment1'])}" + \
                           f"{int(self.space.iloc[i]['Treatment2'])}"
                    new_death.append(
                        np.random.binomial(1, cond_probs_dict[find]))

                else:
                    new_death.append(np.random.binomial(1, 0.05))

            self.space['Death'] = new_death
            self.space_corr_mtx = self.space.corr()

    def add_correlated_symptom_with(
            self,
            explanatory_label='Vaccine1',  # 'Covid-Positive' or 'Treatment1'
            response_label='Fever',
            p=0.8
    ):
        new_ = []
        for i in range(len(self.space)):
            if self.space.iloc[i][explanatory_label] == 1:
                new_.append(np.random.binomial(1, p))
            else:
                new_.append(np.random.binomial(1, 0.05))
        self.space[response_label] = new_
        self.space_corr_mtx = self.space.corr()
