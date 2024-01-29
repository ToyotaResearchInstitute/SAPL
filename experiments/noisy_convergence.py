#!/usr/bin/python3
"""
Safe Active Preference Learning (APL) Experiment Script

This script conducts experiments to test the convergence properties of
the Safe Active Preference Learning (SAPL) framework when answers are noisy.
It uses trajectories and robustness functions to analyze the
behavior of the framework in various scenarios.

This experiment is in Section VII.A.(c) and titled "Resilience to noisy answers"

The experiment results are saved in a CSV file named
'{experiment}_noisy_convergence_analysis.csv'.

Command-line Arguments:
    - --no_samples: Number of samples for weight sets (default: 1000).
    - --terminating_condition: Probability limit to end the learning framework
                               (default: 0.99).
    - --no_questions: Number of questions to be asked to the user (default: 200).
    - --experiment: Type of experiment
                    (Options: 'pedestrian', 'overtake'; default: 'overtake').
    - --repetition: Number of times the test will be repeated (default: 100).

Example:
    $ python3 noisy_convergence.py
    --no_samples 1000 --terminating_condition 0.99
    --experiment overtake --repetition 100

Author: Ruya Karagulle
Date: September 2023
"""

import os
import sys
import pickle

import numpy as np
import random
import argparse
import pandas as pd

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

# codebase specific libraries
from APL_utils import *  # NOQA
from preprocess_utils import *  # NOQA

random.seed(0)


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
        default=200,
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

    u = 0.4
    rob_diff_bound = -np.log(u / (1 - u))
    output = []
    for i in range(repetition):
        random.seed(i)
        w_final = random.randint(0, no_samples)
        formula.set_weights(
            signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True, seed=i
        )

        apl_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
        )

        output.append(
            apl_instance.noisy_convergence(threshold_probability, no_questions, w_final)
        )

    df = pd.DataFrame(output)
    df.to_csv(f"./results/{experiment}_noisy_convergence_analysis.csv", encoding="utf-8")
    return df


def save_stats(dfs, repetition):
    df = pd.concat(dfs)
    pedestrian_converged_indices = np.array(
        [df["converged_w_index"].iloc[k] for k in range(repetition)]
    )
    pedestrian_correct_indices = np.array(
        [df["correct_w_index"].iloc[k] for k in range(repetition)]
    )
    pedestrian_train_agreement = 100 * np.array(
        [df["no_questions_agreed_of_asked"].iloc[k] for k in range(repetition)]
    )
    pedestrian_all_agreement = 100 * np.array(
        [df["no_questions_agreed"].iloc[k] for k in range(repetition)]
    )

    overtake_converged_indices = np.array(
        [df["converged_w_index"].iloc[k + repetition] for k in range(repetition)]
    )
    overtake_correct_indices = np.array(
        [df["correct_w_index"].iloc[k + repetition] for k in range(repetition)]
    )
    overtake_train_agreement = 100 * np.array(
        [
            df["no_questions_agreed_of_asked"].iloc[k + repetition]
            for k in range(repetition)
        ]
    )
    overtake_all_agreement = 100 * np.array(
        [df["no_questions_agreed"].iloc[k + repetition] for k in range(repetition)]
    )

    pedestrian_convergence_rate = (
        100
        * sum(pedestrian_converged_indices == pedestrian_correct_indices)
        / pedestrian_converged_indices.shape[0]
    )

    overtake_convergence_rate = (
        100
        * sum(overtake_converged_indices == overtake_correct_indices)
        / overtake_converged_indices.shape[0]
    )

    table = pd.DataFrame()
    table["Scenario"] = ["Pedestrian", "Overtake"]
    table["Convergence Rate"] = [pedestrian_convergence_rate, overtake_convergence_rate]
    table["Minimum Train Agreement"] = [
        np.min(pedestrian_train_agreement),
        np.min(overtake_train_agreement),
    ]
    table["Maximum Train Agreement"] = [
        np.max(pedestrian_train_agreement),
        np.max(overtake_train_agreement),
    ]
    table["Minimum Overall Agreement"] = [
        np.min(pedestrian_all_agreement),
        np.min(overtake_all_agreement),
    ]
    table["Maximum Overall Agreement"] = [
        np.max(pedestrian_all_agreement),
        np.max(overtake_all_agreement),
    ]

    print(table)
    table.to_csv("./results/SAPL_table2.csv", encoding="utf-8")


def main():
    """
    Main entry point for the script.
    Read data, initialize experiment parameters, and call apl_experiment.

    """
    args = create_arguments()
    repeatability_evaluation = args.repeatability_evaluation

    if repeatability_evaluation:  # reproduce experiment in Table1 of the paper
        no_questions = 12
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
        save_stats(dfs, repetition)
    else:
        no_samples = args.no_samples
        threshold_probability = args.terminating_condition
        no_questions = args.no_questions
        experiment = args.experiment
        repetition = args.repetition

        data_name = f"./data/{experiment}_trajectories.pkl"
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
