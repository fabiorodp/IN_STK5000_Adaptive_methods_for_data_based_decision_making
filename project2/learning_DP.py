import pandas as pd
import numpy as np


def pi(x, f_function='identity'):
    r"""
    Defines a distribution over responses a given the data x and the query q.

    Notations
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_{n} := individuals.
    k_{i} := attributes.

    Formula
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    \pi(a | x, q)

    Retuns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a := response given the data x and the query q.
    """
    sensitivity = 1
    epsilon = 0.1

    if f_function == 'identity':
        y = x

    elif f_function == 'mean':
        y = np.mean(x)

    elif f_function == 'sum':
        y = np.sum(x)

    else:
        raise ValueError("ERROR: f_function not given correctly.")

    return y + np.random.laplace(0, sensitivity / epsilon)


def f(X, j):
    result = 0
    for i in range(len(X)):
        if i != j:
            if i == 0:
                result = X[i]
                continue
            elif i == 1:
                result = X[i]
                continue

            result *= X[i]
    return result


if __name__ == '__main__':
    X = np.array([50, 10, 90, 40, 35, 17, 98, 23, 45, 65])
    X_prime = np.array([50, 10, 90, 40, 35, 1, 98, 23, 45, 65])

    r = f(X, 0)

    income_data = \
        'https://raw.githubusercontent.com/dhesse/IN-STK5000-Autumn21/' \
        'main/income_data.csv'

    df = pd.read_csv(income_data, skipinitialspace=True)

    a = pi(x=df['age'], f_function='identity')
