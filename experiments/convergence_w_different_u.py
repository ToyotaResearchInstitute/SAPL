#!/usr/bin/python3
"""
Safe Active Preference Learning (APL) Experiment Script

This script conducts experiments to test the convergence properties of
the Safe Active Preference Learning (SAPL) framework with different u bounds.
It uses trajectories and robustness functions to analyze the
behavior of the framework in various scenarios.

This experimentis in Section VII.A.(b) and titled
"The effect of different probabilistic bounds u"

The experiment results are saved in a CSV file named
'{experiment}_different_u_convergence_analysis.csv'.

Command-line Arguments:
    - --no_samples: Number of samples for weight sets (default: 1000).
    - --terminating_condition: Probability limit to end the learning framework
                               (default: 0.99).
    - --no_questions: Maximum number of questions to ask (default: 20).
    - --experiment: Type of experiment
                    (Options: 'pedestrian', 'overtake'; default: 'overtake').
    - --repetition: Number of times the test will be repeated (default: 100).

Example:
    $ python convergence_w_different_u.py
    --no_samples 1000 --terminating_condition 0.99
    --no_questions 20 --experiment overtake --repetition 100

Author: Ruya Karagulle
Date: September 2023
"""
import random
import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc("font", family="serif")
# plt.rc('text', usetex=True)


cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

# codebase specific libraries
from APL_utils import *  # NOQA
from preprocess_utils import *  # NOQA


def create_arguments():
    """
    Define command-line arguments for the experiment configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="The program tests the convergence properties"
    )
    parser.add_argument(
        "--no_samples",
        default=1000,
        type=int,
        help="Nummber of samples for weight sets",
    )
    parser.add_argument(
        "--terminating_condition",
        default=0.99,
        type=float,
        help="Limit probability to reach to end the learning framework.",
    )
    parser.add_argument(
        "--no_questions",
        default=20,
        type=int,
        help=" Number of questions to be asked to user.",
    )
    parser.add_argument(
        "--experiment",
        default="overtake",
        type=str,
        help="Experiment type. Options: pedestrian, overtake ",
    )
    parser.add_argument(
        "--repetition",
        default=100,
        type=int,
        help="number of times the test will be repeated.",
    )
    parser.add_argument(
        "--repeatability_evaluation",
        action="store_true",
        help="Flag to run the experiment on Table1 of the paper. ",
    )
    return parser.parse_args()


def apl_experiment(
    signals: tuple,
    formula: WSTL.WSTL_Formula,
    no_samples: int,
    threshold_probability: float,
    no_questions: int,
    repetition: int,
    experiment: str,
):
    """
    Conduct the SAPL experiment with specified parameters.

    Args:
        signals: Preprocessed signals.
        formula: Scaled WSTL formula.
        no_samples: Number of samples for weight sets.
        threshold_probability: Probability limit to end the learning framework.
        no_questions: Number of questions to be asked in the experiment.
        repetition: Number of times the test will be repeated.
        experiment: Type of experiment (e.g., 'overtake', 'pedestrian').
    """

    output = []

    w_final = random.randint(0, no_samples)
    formula.set_weights(
        signals, w_range=[0.1, 1.1], no_samples=no_samples, seed=0, random=True
    )

    for i in range(repetition):
        random.seed(i)

        u = (i + 1) * 0.5 / repetition
        rob_diff_bound = -np.log(u / (1 - u))
        apl_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
        )
        output.append(
            apl_instance.convergence(threshold_probability, no_questions, w_final)
        )

    df = pd.DataFrame(output)
    df.to_csv(
        f"./results/{experiment}_different_u_convergence_analysis.csv", encoding="utf-8"
    )
    return df


def save_stats(repetition):
    colors = ["darkorange", "teal", "crimson"]
    fig, axs = plt.subplots(2, 1, figsize=(17, 8))
    alph = 0.8
    f = 26
    experiment = ["pedestrian", "overtake"]
    for i, e in enumerate(experiment):
        df = pd.read_csv(f"./results/{e}_different_u_convergence_analysis.csv")
        no_questions = df["no_questions_asked"]
        correct_ind = df["correct_w_index"]
        converged_ind = df["converged_w_index"]
        non_converged_indices = [
            i for i, x in enumerate(correct_ind != converged_ind) if x
        ]
        non_converged_u = (
            (np.array([i for i, x in enumerate(correct_ind != converged_ind) if x]) + 1)
            * 0.5
            / 100
        )
        max_of_small_u = [
            non_converged_u[i]
            for i, x in enumerate(
                non_converged_indices == np.arange(len(non_converged_indices))
            )
            if x
        ][-1]
        min_of_big_u = [
            non_converged_u[i]
            for i, x in enumerate(
                non_converged_indices
                == np.flip(
                    np.arange(
                        max(non_converged_indices),
                        max(non_converged_indices) - len(non_converged_indices),
                        -1,
                    )
                )
            )
            if x
        ][0]

        train_agreement = df["no_questions_agreed_of_asked"]
        all_agreement = df["no_questions_agreed"]
        u = (np.arange(repetition - 1) + 1) * 0.5 / repetition
        ax_right = axs[i].twinx()  # Create a twin axis sharing the same x-axis

        if i == 0:
            axs[i].plot(
                u,
                train_agreement[:-1],
                color=colors[0],
                alpha=alph,
                linewidth=2,
                label="Agreement for the training set",
            )
            axs[i].plot(
                u,
                all_agreement[:-1],
                color=colors[1],
                alpha=alph,
                linewidth=2,
                label="Agreement for all questions",
            )
            ax_right.plot(
                u, no_questions[:-1], "-.", color=colors[2], alpha=alph, linewidth=2
            )

        else:
            axs[i].plot(u, train_agreement[:-1], color=colors[0], alpha=alph, linewidth=2)
            axs[i].plot(u, all_agreement[:-1], color=colors[1], alpha=alph, linewidth=2)
            ax_right.plot(
                u,
                no_questions[:-1],
                "-.",
                color=colors[2],
                alpha=alph,
                linewidth=2,
                label="Number of questions asked",
            )
        axs[i].set_xlabel("u limit for the likelihood function", fontsize=f)
        ax_right.set_ylabel("Questions", fontsize=f)
        axs[i].fill_between(
            u,
            0.6,
            1,
            where=(u > max_of_small_u) & (u < min_of_big_u),
            alpha=0.35,
            color="pink",
            label="region of convergence to correct weight valuation",
        )
        axs[i].set_ylim([0.6, 1.05])
        ax_right.set_ylim([4, 21])
        ax_right.set_yticklabels([5, 10, 15, 20], fontsize=f)
        axs[i].set_yticklabels([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=f)
        axs[i].set_xlim([0, 0.5])
        axs[i].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=f)
        ax_right.set_yticks([5, 10, 15, 20])
        axs[i].set_ylabel("User Agreement", fontsize=f)
        # axs[i].set_title(f'the {e} scenario',  pad=20, loc='right')
        axs[i].set_title(f"the {e} scenario", fontsize=f, loc="left")

        handles1, labels1 = axs[0].get_legend_handles_labels()
        handles2, labels2 = ax_right.get_legend_handles_labels()

    fig.legend(
        handles=handles1 + handles2,
        labels=labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=f - 1,
    )
    axs[0].set_position([0.13, 0.62, 0.77, 0.385])  # [left, bottom, width, height]
    axs[1].set_position([0.13, 0.12, 0.77, 0.385])  # [left, bottom, width, height]

    # axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol= 2, fontsize= f)

    plt.tight_layout()
    plt.savefig("./results/u_bound.png")


def main():
    args = create_arguments()
    repeatability_evaluation = args.repeatability_evaluation

    if repeatability_evaluation:  # reproduce experiment in Table1 of the paper
        no_questions = 20
        no_samples = 1000
        threshold_probability = 0.99
        repetition = 100
        dfs = []
        for exp in ["pedestrian", "overtake"]:
            data_name = f"./data/{exp}_trajectories.pkl"
            with open(data_name, "rb") as f:
                data = pickle.load(f)

            processed_signals = get_signals(data, exp)
            phi = get_formula(processed_signals, exp)

            df = apl_experiment(
                processed_signals,
                phi,
                no_samples,
                threshold_probability,
                no_questions,
                repetition,
                exp,
            )
            dfs.append(df)
        save_stats(repetition)

    else:
        no_samples = args.no_samples
        threshold_probability = args.terminating_condition
        no_questions = args.no_questions
        experiment = args.experiment
        repetition = args.repetition

        data_name = f"./data/{args.experiment}_trajectories.pkl"
        with open(data_name, "rb") as f:
            data = pickle.load(f)

        processed_signals = get_signals(data, experiment)
        phi = get_formula(processed_signals, experiment)

        df = apl_experiment(
            processed_signals,
            phi,
            no_samples,
            threshold_probability,
            no_questions,
            repetition,
            experiment,
        )


if __name__ == "__main__":
    main()
