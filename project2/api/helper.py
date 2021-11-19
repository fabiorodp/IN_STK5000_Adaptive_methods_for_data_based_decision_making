import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def fairness_unifying_view(table):
    z_labels = ['leq30', 'bt3060', 'geq60', 'female', 'male',
                'bt0k10k', 'geq10k']
    arr = np.zeros(shape=(len(z_labels), len(z_labels)))

    tb_df = pd.DataFrame(arr, index=z_labels, columns=z_labels)
    for r in range(len(z_labels)):
        for c in range(len(z_labels)):
            tb_df.iloc[r, c] = max(abs(table[:, r] - table[:, c]))

    print(tb_df)
    return tb_df


def fairness_calibration(pl):
    all_not_vaccinated, all_vaccinated, all = pl.vstack_all()
    tables_py_a, tables_py_az = [], []

    for d, x in zip(
            [all, all_not_vaccinated, all_vaccinated],
            ['all vaccinated population', 'vaccinated by us',
             'vaccinated before']
    ):
        ages = []
        for a in range(len(d['Age'])):
            if d.iloc[a, 10] <= 30:
                ages.append('leq30')
            elif (d.iloc[a, 10] > 30) and (d.iloc[a, 10] < 60):
                ages.append('bt3060')
            else:
                ages.append('geq60')

        genders = []
        for a in range(len(d['Income'])):
            if d.iloc[a, 11] == 0:
                genders.append('female')
            else:
                genders.append('male')

        incomes = []
        for a in range(len(d['Income'])):
            if d.iloc[a, 12] < 10000:
                incomes.append('bt0k10k')
            else:
                incomes.append('geq10k')

        vaccines = []
        for a in range(len(d)):
            if d.iloc[a, -3] == 1:
                vaccines.append(1)
            elif d.iloc[a, -2] == 1:
                vaccines.append(2)
            elif d.iloc[a, -1] == 1:
                vaccines.append(3)

        df = pd.DataFrame(ages, columns=['Age'])
        df['Gender'] = genders
        df['Income'] = incomes
        df['Vaccine1'] = d['Vaccine1'].values
        df['Vaccine2'] = d['Vaccine2'].values
        df['Vaccine3'] = d['Vaccine3'].values
        df['Death'] = d['Death'].values
        df['Vaccines'] = vaccines

        arr = np.zeros(shape=(1, 3), dtype=np.float)
        py_a = pd.DataFrame(
            arr,
            columns=['Vaccine1', 'Vaccine2', 'Vaccine3'],
            index=['Dead']
        )

        crosstable = pd.crosstab(
            index=df['Death'],
            columns=df['Vaccines'],
            margins=True
        )

        py_a.iloc[0, 0] = crosstable.iloc[1, 0] / crosstable.iloc[2, 0]
        py_a.iloc[0, 1] = crosstable.iloc[1, 1] / crosstable.iloc[2, 1]
        py_a.iloc[0, 2] = crosstable.iloc[1, 2] / crosstable.iloc[2, 2]

        print(f"Calibration's table with P(y|a) (space: {x})")
        print(py_a)
        tables_py_a.append(py_a)

        z_labels = ['leq30', 'bt3060', 'geq60', 'female', 'male',
                    'bt0k10k', 'geq10k']

        arr = np.zeros(shape=(3, len(z_labels)), dtype=np.float)
        py_az = pd.DataFrame(
            arr,
            columns=z_labels,
            index=['Vaccine1', 'Vaccine2', 'Vaccine3']
        )

        for r, v in enumerate(['Vaccine1', 'Vaccine2', 'Vaccine3']):
            for c, z in enumerate(z_labels):

                # r = 0
                # v = 'Vaccine1'
                # c = 3
                # z = 'female'

                if z in ['leq30', 'bt3060', 'geq60']:
                    paz = df[(df[v] == 1) & (df['Age'] == z)].shape[0]

                    pyaz = df[
                        (df['Death'] == 1) & (df[v] == 1) & (df['Age'] == z)
                        ].shape[0]

                    py_az.iloc[r, c] = pyaz / paz

                elif z in ['female', 'male']:
                    # z = 'female'
                    paz = df[(df[v] == 1) & (df['Gender'] == z)].shape[0]

                    pyaz = df[
                        (df['Death'] == 1) & (df[v] == 1) & (df['Gender'] == z)
                        ].shape[0]

                    py_az.iloc[r, c] = pyaz / paz

                elif z in ['bt0k10k', 'geq10k']:
                    paz = df[(df[v] == 1) & (df['Income'] == z)].shape[0]

                    pyaz = df[
                        (df['Death'] == 1) & (df[v] == 1) & (df['Income'] == z)
                        ].shape[0]

                    py_az.iloc[r, c] = pyaz / paz

        print(f"Calibration's table with P(y|az) (space: {x})")
        print(py_az)
        tables_py_az.append(py_az)

    return tables_py_a, tables_py_az


def fairness_simple_probabilities(pl):
    all_not_vaccinated, all_vaccinated, all = pl.vstack_all()
    tables_pa_z = []

    for d, x in zip(
            [all, all_not_vaccinated, all_vaccinated],
            ['all vaccinated population', 'vaccinated by us',
             'vaccinated before']
    ):
        ages = []
        for a in range(len(d['Age'])):
            if d.iloc[a, 10] <= 30:
                ages.append('leq30')
            elif (d.iloc[a, 10] > 30) and (d.iloc[a, 10] < 60):
                ages.append('bt3060')
            else:
                ages.append('geq60')

        incomes = []
        for a in range(len(d['Income'])):
            if d.iloc[a, 12] < 10000:
                incomes.append('bt0k10k')
            else:
                incomes.append('geq10k')

        df = pd.DataFrame(ages, columns=['Age'])
        df['Gender'] = d['Gender'].values
        df['Income'] = incomes
        df['Vaccine1'] = d['Vaccine1'].values
        df['Vaccine2'] = d['Vaccine2'].values
        df['Vaccine3'] = d['Vaccine3'].values
        df['Death'] = d['Death'].values

        a_labels = ['Vaccine1', 'Vaccine2', 'Vaccine3']
        z_labels = ['leq30', 'bt3060', 'geq60', 'female', 'male', 'bt0k10k',
                    'geq10k']
        group = np.zeros(shape=(len(a_labels), len(z_labels)), dtype=np.float)
        pa_z = pd.DataFrame(group, columns=z_labels, index=a_labels)

        for r, a in enumerate(a_labels):
            crosstable0 = pd.crosstab(
                index=df[a],
                columns=df['Age'],
                margins=True
            )
            crosstable1 = pd.crosstab(
                index=df[a],
                columns=df['Gender'],
                margins=True
            )
            crosstable1.columns = ['female', 'male', 'All']

            crosstable2 = pd.crosstab(
                index=df[a],
                columns=df['Income'],
                margins=True
            )

            for c, z in enumerate(z_labels):
                if z in ['leq30', 'bt3060', 'geq60']:
                    pa_z.iloc[r, c] = crosstable0[z][1] / crosstable0[z][2]
                elif z in ['female', 'male']:
                    pa_z.iloc[r, c] = crosstable1[z][1] / crosstable1[z][2]
                else:
                    pa_z.iloc[r, c] = crosstable2[z][1] / crosstable2[z][2]

        tables_pa_z.append(pa_z)
        # printing the probability tables
        print(f"Table with P(a|z) (space: {x})")
        print(pa_z)

    groups = np.zeros(shape=(3, 3))
    pa = pd.DataFrame(
        groups,
        columns=['Vaccine1', 'Vaccine2', 'Vaccine3'],
        index=['all', 'vaccinated by us', 'vaccinated before']
    )

    for r, (v, d) in enumerate(
            zip(
                ['all', 'vaccinated by us', 'vaccinated before'],
                [all, all_not_vaccinated, all_vaccinated]
            )
    ):
        for c, v in enumerate(['Vaccine1', 'Vaccine2', 'Vaccine3']):
            pa.iloc[r, c] = d[v].sum() / d.shape[0]

    print(pa)
    return tables_pa_z, pa


def plot_dist_gender(pl):
    all_not_vaccinated, all_vaccinated, all = pl.vstack_all()
    plt.hist(
        all['Gender'],
        color='lime',
        bins=40,
        label="All vaccinated population",
        alpha=0.7,
    )
    plt.hist(
        all_not_vaccinated['Gender'],
        color='pink',
        bins=40,
        label="Vaccinated by us",
        alpha=1,
    )
    plt.hist(
        all_vaccinated['Gender'],
        color='orange',
        bins=40,
        alpha=0.5,
        label="Vaccinated before"
    )

    plt.title("Gender distributions in the historical data", size=13)
    plt.xlabel("Gender", size=13)
    plt.ylabel('Count', size=13)
    plt.grid()
    plt.legend()
    plt.show()


def plot_dist_income(pl):
    all_not_vaccinated, all_vaccinated, all = pl.vstack_all()
    plt.hist(
        all['Income'],
        color='lime',
        bins=40,
        label="All vaccinated population",
        alpha=0.7,
    )
    plt.hist(
        all_not_vaccinated['Income'],
        color='pink',
        bins=40,
        label="Vaccinated by us",
        alpha=1,
    )
    plt.hist(
        all_vaccinated['Income'],
        color='orange',
        bins=40,
        alpha=0.5,
        label="Vaccinated before"
    )

    plt.title("Income distributions in the historical data", size=13)
    plt.xlabel("Income", size=13)
    plt.ylabel('Count', size=13)
    plt.grid()
    plt.legend()
    plt.xlim([0, 60000])
    plt.show()


def plot_dist_age(pl):
    all_not_vaccinated, all_vaccinated, all = pl.vstack_all()
    plt.hist(
        all['Age'],
        color='lime',
        bins=40,
        label="All vaccinated population",
        alpha=0.7,
    )
    plt.hist(
        all_not_vaccinated['Age'],
        color='pink',
        bins=40,
        label="Vaccinated by us",
        alpha=1,
    )
    plt.hist(
        all_vaccinated['Age'],
        color='orange',
        bins=40,
        alpha=0.5,
        label="Vaccinated before"
    )

    plt.title("Age distributions in the historical data", size=13)
    plt.xlabel("Age", size=13)
    plt.ylabel('Count', size=13)
    plt.grid()
    plt.legend()
    plt.show()


def plot_expected_utilities(pl):
    plt.plot(
        [i for i in range(2, pl.vaccination_stage + 1)],
        pl.ML_expected_utilities,
        color='green',
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=5,
        label="ML Policy"
    )
    plt.plot(
        [i for i in range(1, pl.vaccination_stage + 1)],
        [np.mean(pl.random_expected_utilities) for _ in
         range(1, pl.vaccination_stage + 1)],
        color='red',
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=5,
        label="Mean for Random Policy"
    )
    plt.plot(
        [i for i in range(1, pl.vaccination_stage + 1)],
        pl.observed_expected_utilities,
        color='orange',
        marker='o',
        linestyle='dashed',
        linewidth=2,
        markersize=5,
        label="Observed Deaths"
    )
    plt.title('Expected utilities for ML and Random vaccination policies')
    plt.xlabel('Vaccination stages')
    plt.ylabel('Estimation for the number of deaths')
    plt.legend()
    plt.grid()
    plt.show()


def fairness_barplot(pl, vaccination_stage, plot_for='Age'):
    if plot_for == 'Age':
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
                pl.l30_v1[vaccination_stage],
                pl.bt3060_v1[vaccination_stage],
                pl.g60_v1[vaccination_stage],
                pl.l30_v2[vaccination_stage],
                pl.bt3060_v2[vaccination_stage],
                pl.g60_v2[vaccination_stage],
                pl.l30_v3[vaccination_stage],
                pl.bt3060_v3[vaccination_stage],
                pl.g60_v3[vaccination_stage],
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Ages",
            data=df1,
            palette="Accent_r",
        )

        plt.title(
            f"Vaccination fairness among ages (stage:{vaccination_stage})",
            size=12
        )

        plt.ylabel("Frequency", size=12)
        plt.xlabel("Vaccines", size=12)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.grid()
        plt.show()

    elif plot_for == 'Gender':
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
                pl.female_v1[vaccination_stage],
                pl.male_v1[vaccination_stage],
                pl.female_v2[vaccination_stage],
                pl.male_v2[vaccination_stage],
                pl.female_v3[vaccination_stage],
                pl.male_v3[vaccination_stage]
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Genders",
            data=df1,
            palette="Accent_r",
        )

        plt.title(
            f"Vaccination fairness between "
            f"genders (stage:{vaccination_stage})",
            size=12
        )

        plt.ylabel("Frequency", size=12)
        plt.xlabel("Vaccines", size=12)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        plt.grid()
        plt.show()

    elif plot_for == 'Income':
        df1 = pd.DataFrame({
            'Vaccines': [
                'Vaccine1', 'Vaccine1',
                'Vaccine2', 'Vaccine2',
                'Vaccine3', 'Vaccine3',
            ],
            'Incomes': [
                "0k10k", "geq10k",
                "0k10k", "geq10k",
                "0k10k", "geq10k",
            ],
            'Frequency': [
                pl.i0k10k_v1[vaccination_stage],
                pl.geq10k_v1[vaccination_stage],
                pl.i0k10k_v2[vaccination_stage],
                pl.geq10k_v2[vaccination_stage],
                pl.i0k10k_v3[vaccination_stage],
                pl.geq10k_v3[vaccination_stage],
            ],
        })

        sns.barplot(
            x='Vaccines',
            y='Frequency',
            hue="Incomes",
            data=df1,
            palette="Accent_r",
        )

        plt.title(
            f"Vaccination fairness among incomes (stage:{vaccination_stage})",
            size=12
        )

        plt.ylabel("Frequency", size=12)
        plt.xlabel("Vaccines", size=12)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5)
        plt.grid()
        plt.show()


def fairness_lineplot(pl, plot_for='Age'):
    if plot_for == "Age":
        for v, b in zip(
                ["Vaccine1", "Vaccine2", "Vaccine3"],
                [
                    [pl.l30_v1[1:], pl.bt3060_v1[1:], pl.g60_v1[1:]],
                    [pl.l30_v2[1:], pl.bt3060_v2[1:], pl.g60_v2[1:]],
                    [pl.l30_v3[1:], pl.bt3060_v3[1:], pl.g60_v3[1:]],
                ]
        ):
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[0],
                color='green',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Up to 30 years"
            )
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[1],
                color='red',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="30 and 60 years"
            )
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[2],
                color='orange',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Greater than 60 years"
            )

            plt.title(f'Fairness for age category ({v})')
            plt.xlabel('Vaccination stages')
            plt.ylabel('Number of people Vaccinated')
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
            plt.grid()
            plt.show()

    if plot_for == "Gender":
        for v, b in zip(
                ["Vaccine1", "Vaccine2", "Vaccine3"],
                [
                    [pl.female_v1[1:], pl.male_v1[1:]],
                    [pl.female_v2[1:], pl.male_v2[1:]],
                    [pl.female_v3[1:], pl.male_v3[1:]],
                ]
        ):
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[0],
                color='green',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Female"
            )
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[1],
                color='red',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Male"
            )

            plt.title(f'Fairness for gender category ({v})')
            plt.xlabel('Vaccination stages')
            plt.ylabel('Number of people Vaccinated')
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
            plt.grid()
            plt.show()

    if plot_for == "Income":
        for v, b in zip(
                ["Vaccine1", "Vaccine2", "Vaccine3"],
                [
                    [pl.i0k10k_v1[1:], pl.geq10k_v1[1:]],
                    [pl.i0k10k_v2[1:], pl.geq10k_v2[1:]],
                    [pl.i0k10k_v3[1:], pl.geq10k_v3[1:]],
                ]
        ):
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[0],
                color='green',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Female"
            )
            plt.plot(
                [i for i in range(2, pl.vaccination_stage + 1)],
                b[1],
                color='red',
                marker='o',
                linestyle='dashed',
                linewidth=2,
                markersize=5,
                label="Male"
            )

            plt.title(f'Fairness for income category ({v})')
            plt.xlabel('Vaccination stages')
            plt.ylabel('Number of people Vaccinated')
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
            plt.grid()
            plt.show()
