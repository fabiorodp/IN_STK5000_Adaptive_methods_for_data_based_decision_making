## Comborbidities:
## Asthma, Obesity,  Smoking, Diabetes, Heart diseae, Hypertension
## Symptom list: Covid-Recovered, Covid-Positive, Taste, Fever, Headache,
# Pneumonia, Stomach, Myocarditis, Blood-Clots, Death
## Mild symptoms: Taste, Fever, Headache, Stomach
## Critical symptoms: Pneumonia, Myocarditis, Blood-Clots
import numpy as np
import pickle


class Person:
    def __init__(self, pop):
        self.genes = np.random.choice(2, size=pop.n_genes)
        self.gender = np.random.choice(2, 1)
        self.age = np.random.gamma(3, 11)
        self.age_adj = self.age / 100  # age affects everything
        self.income = np.random.gamma(1, 10000)
        self.comorbidities = [0] * pop.n_comorbidities
        self.comorbidities[0] = pop.asthma
        self.comorbidities[1] = pop.obesity * self.age_adj
        self.comorbidities[2] = pop.smoking
        self.diab = pop.diabetes + self.comorbidities[1] * 0.5
        self.HT = pop.htension + self.comorbidities[2] * 0.5
        self.comorbidities[3] = self.diab
        self.comorbidities[4] = pop.heart * self.age_adj
        self.comorbidities[5] = self.HT * self.age_adj

        for i in range(pop.n_comorbidities):
            if (np.random.uniform() < self.comorbidities[i]):
                self.comorbidities[i] = 1
            else:
                self.comorbidities[i] = 0
        self.symptom_baseline = np.array(
            [pop.historical_prevalence, pop.prevalence, 0.3, 0.2, 0.05, 0.2,
             0.02, 0.05, 0.2, 0.1]);
        self.symptom_baseline = np.array(
            np.matrix(self.genes) * pop.G).flatten() * self.symptom_baseline
        self.symptom_baseline[0] = pop.historical_prevalence;
        self.symptom_baseline[1] = pop.prevalence;
        if (self.gender == 1):
            self.symptom_baseline[8] += 0.01
        else:
            self.symptom_baseline[7] += 0.01
            self.symptom_baseline[9] += 0.01
        ## previous infection

        # Initially no symptoms apart from Covid+/CovidPre
        self.symptoms = [0] * pop.n_symptoms
        if (np.random.uniform() <= self.symptom_baseline[0]):
            self.symptoms[0] = 1
        if (np.random.uniform() <= self.symptom_baseline[1]):
            self.symptoms[1] = 1
        self.vaccines = [0] * pop.n_vaccines

    # use vaccine = -1 if no vaccine is given
    def vaccinate(self, vaccine, pop):
        ## Vaccinated
        print(vaccine, pop.n_vaccines)
        if (vaccine >= 0):
            self.vaccines[vaccine] = 1
            self.symptom_baseline[1] *= pop.baseline_efficacy[vaccine]

        if (vaccine >= 0 and self.symptoms[1] == 1):
            self.symptom_baseline[[2, 3, 4, 6]] *= pop.mild_efficacy[vaccine]
            self.symptom_baseline[[5, 7, 8]] *= pop.critical_efficacy[vaccine]
            self.symptom_baseline[9] *= pop.death_efficacy[vaccine]

        if (self.symptoms[0] == 1):
            self.symptom_baseline *= 0.5

        # baseline symptoms of non-covid patients
        if (self.symptoms[0] == 0 and self.symptoms[1] == 0):
            self.symptom_baseline = np.array(
                [0, 0, 0.001, 0.01, 0.02, 0.002, 0.005, 0.001, 0.002, 0.0001]);
            ## Common side-effects
            if (vaccine == 1):
                self.symptom_baseline[8] += 0.01
                self.symptom_baseline[9] += 0.001
            if (vaccine == 2):
                self.symptom_baseline[7] += 0.01
            if (vaccine >= 0):
                self.symptom_baseline[3] += 0.2
                self.symptom_baseline[4] += 0.1

        # 10% long covid sufferers
        if (self.symptoms[0] == 1 and self.symptoms[1] == 0):
            self.symptom_baseline *= 2

        # genetic factors
        self.symptom_baseline = np.array(
            np.matrix(self.genes) * pop.G).flatten() * self.symptom_baseline
        # print("V:", vaccine, symptom_baseline)
        for s in range(2, pop.n_symptoms):
            if (np.random.uniform() < self.symptom_baseline[s]):
                self.symptoms[s] = 1


class Population:
    def __init__(self, n_genes, n_vaccines, n_treatments):
        self.n_genes = n_genes
        self.n_comorbidities = 6;
        self.n_symptoms = 10
        self.n_vaccines = n_vaccines
        self.n_treatments = n_treatments
        self.G = np.random.uniform(size=[self.n_genes, self.n_symptoms])
        self.G /= sum(self.G)
        self.A = np.random.uniform(size=[self.n_treatments, self.n_symptoms])
        self.asthma = 0.08
        self.obesity = 0.3
        self.smoking = 0.2
        self.diabetes = 0.1
        self.heart = 0.15
        self.htension = 0.3
        self.baseline_efficacy = [0.5, 0.6, 0.7]
        self.mild_efficacy = [0.6, 0.7, 0.8]
        self.critical_efficacy = [0.8, 0.75, 0.85]
        self.death_efficacy = [0.9, 0.95, 0.9]
        self.vaccination_rate = [0.7, 0.1, 0.1, 0.1]
        self.prevalence = 0.1
        self.historical_prevalence = 0.1

    ## Generates data with the following structure:
    ## X: characteristics before treatment, including whether or not they were vaccinated
    ## The generated population may already be vaccinated.
    def generate(self, n_individuals):
        X = np.zeros([n_individuals,
                      3 + self.n_genes + self.n_comorbidities + self.n_vaccines + self.n_symptoms])
        Y = np.zeros([n_individuals, self.n_treatments, self.n_symptoms])
        self.persons = []
        for t in range(n_individuals):
            person = Person(self)
            vaccine = np.random.choice(4, p=self.vaccination_rate) - 1
            person.vaccinate(vaccine, self)
            self.persons.append(person)
            x_t = np.concatenate(
                [person.symptoms, [person.age, person.gender, person.income],
                 person.genes, person.comorbidities, person.vaccines])
            X[t, :] = x_t
        self.X = X
        return X

    def vaccinate(self, person_index, vaccine):
        self.persons[person_index].vaccinate(vaccine, self)
        return self.persons[person_index].symptoms

    def treat(self, person_index, treatment):
        treatments = np.zeros(self.n_treatments)
        treatments[treatment] = 1
        r = np.array(treatment * self.A).flatten()
        result = np.zeros(self.n_symptoms)
        for k in range(self.n_symptoms):
            if (k <= 1):
                result[k] = self.X[person_index, k]
            else:
                if (np.random.uniform() < r[k]):
                    result[k] = 0
                else:
                    result[k] = self.X[person_index, k]
        return result

    def get_features(self, person_index):
        x_t = np.concatenate([self.persons[t].symptoms,
                              [self.persons[t].age, self.persons[t].gender,
                               self.persons[t].income], self.persons[t].genes,
                              self.persons[t].comorbidities,
                              self.persons[t].vaccines])
        return x_t

    ## Treats a population
    def treatment(self, X, policy):
        treatments = np.zeros([X.shape[0], self.n_treatments])
        result = np.zeros([X.shape[0], self.n_symptoms])
        for t in range(X.shape[0]):
            # print ("X:", result[t])
            treatments[t][policy.get_action(X[t])] = 1
            r = np.array(np.matrix(treatments[t]) * self.A).flatten()
            for k in range(self.n_symptoms):
                if (k <= 1):
                    result[t, k] = X[t, k]
                else:
                    if (np.random.uniform() < r[k]):
                        result[t, k] = 0
                    else:
                        result[t, k] = X[t, k]
            ##print("X:", X[t,:self.n_symptoms] , "Y:",  result[t])
        return treatments, result


# main
if __name__ == "__main__":
    import pandas

    try:
        import policy as policy
    except:
        import project2.src.covid.policy as policy

    n_symptoms = 10
    n_genes = 128
    n_vaccines = 3
    n_treatments = 4
    pop = Population(n_genes, n_vaccines, n_treatments)
    n_observations = 1000
    X_observation = pop.generate(n_observations)
    pandas.DataFrame(X_observation).to_csv('observation_features.csv',
                                           header=False, index=False)
    n_treated = 1000
    X_treatment = pop.generate(n_treated)
    X_treatment = X_treatment[X_treatment[:, 1] == 1]
    print("Generating treatment outcomes")
    a, y = pop.treatment(X_treatment, policy.RandomPolicy(n_treatments))
    pandas.DataFrame(X_treatment).to_csv('treatment_features.csv',
                                         header=False, index=False)
    pandas.DataFrame(a).to_csv('treatment_actions.csv', header=False,
                               index=False)
    pandas.DataFrame(y).to_csv('treatment_outcomes.csv', header=False,
                               index=False)
