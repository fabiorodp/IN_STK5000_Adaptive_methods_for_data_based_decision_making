try:
    from api.users import credentials
    from api.trusted_curator import TrustedCurator
    from api.policy import Policy
except:
    from project2.api.users import credentials
    from project2.api.trusted_curator import TrustedCurator
    from project2.api.policy import Policy


tc = TrustedCurator(
    user='master',
    password='123456789',
    mode='off',
)

pl = Policy(
    n_actions=3,
    action_set=['Vaccine1', 'Vaccine2', 'Vaccine3'],
)

X = tc.get_features()

A = pl.get_actions(
    features=X
)

Y = tc.get_outcomes(
    features=X,
    action=A
)

pl.observe(
    features=X,
    actions=A,
    outcomes=Y
)

pl.get_utility(
    features=X,
    actions=A,
    outcomes=Y
)
