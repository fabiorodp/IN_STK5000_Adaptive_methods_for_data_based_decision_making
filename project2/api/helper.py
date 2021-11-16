import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
