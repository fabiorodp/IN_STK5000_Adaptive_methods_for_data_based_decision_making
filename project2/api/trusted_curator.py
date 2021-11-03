try:
    from users import credentials
except:
    from project2.api.users import credentials


class TrustedCurator:
    def __init__(self, user, password):
        self.user = ''
        if user in credentials.keys():
            if password == credentials[user]:
                self.user = user
            else:
                raise ValueError("ERROR: Wrong user or password.")
        else:
            raise ValueError("ERROR: Wrong user or password.")

    def get_training_data(self):
        """



        :return: X
        """
        pass

    def get_vaccinated_individuals(self):
        """
        Give:
        X', A'

        :return: Y = X x A x Y
        """
        pass


if __name__ == '__main__':
    dp = TrustedCurator(
        user='user1',
        password='12345eu'
    )
