#!/usr/bin/python3

# Copyright 2023 Toyota Research Institute.  All rights reserved.
"""
Safe Active Preference Learning (APL) Experiment Script

This program compares the performance of the query selection method
with random query selection.
It uses trajectories and robustness functions to analyze the
behavior of the framework in various scenarios.

This experimentis in Section VII.A.(a) and titled "The query selection performance"

The experiment results are saved in a CSV file named
'./results/{experiment}_random_query_analysis.csv'.

Command-line Arguments:
    - --no_samples: Number of samples for weight sets (default: 1000).
    - --terminating_condition: Probability limit to end the learning framework
                               (default: 0.99).
    - --experiment: Type of experiment
                    (Options: 'pedestrian', 'overtake'; default: 'overtake').
    - --repetition: Number of times the test will be repeated (default: 100).
    - --repeatability_evaluation: Flag to be used to reproduce datain the paper.

Example:
    $ python3 query_selection_vs_random_ask_comparison.py
    --no_samples 100 --terminating_condition 0.99 --experiment overtake --repetition 10

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
import WSTL  # NOQA


def create_arguments():
    """
    Define command-line arguments for the experiment configuration.
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
        "--experiment",
        default="overtake",
        choices=["pedestrian", "overtake"],
        type=str,
        help="Experiment type. Options: pedestrian, overtake ",
    )
    parser.add_argument(
        "--repeatability_evaluation",
        action="store_true",
        help="Flag to run the experiment on Table1 of the paper. ",
    )
    parser.add_argument(
        "--repetition",
        default=100,
        type=int,
        help="number of times the test will be repeated.",
    )
    return parser.parse_args()


def apl_experiment(
    signals: tuple,
    formula: WSTL.WSTL_Formula,
    no_samples: int,
    threshold_probability: float,
    repetition: int,
    experiment: str,
):
    """
    Conduct the SAPL experiment comparing query selection methods.

    Args:
        signals: Preprocessed signals.
        formula: Scaled WSTL formula.
        no_samples: Number of samples for weight sets.
        threshold_probability: Probability limit to end the learning framework.
        repetition: Number of times the test will be repeated.
        experiment: Type of experiment (e.g., 'overtake', 'pedestrian').
    """
    u = 0.4
    rob_diff_bound = -np.log(u / (1 - u))

    output = []
    for i in range(repetition):
        random.seed(i)
        formula.set_weights(
            signals,
            w_range=[0.1, 1.1],
            no_samples=no_samples,
            random=True,
            seed=i,
        )

        w_final = random.randint(0, no_samples)
        apl_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
        )

        output.append(apl_instance.random_selection(threshold_probability, w_final))
        random_apl = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
        )
        output.append(
            random_apl.random_selection(threshold_probability, w_final, random=True)
        )

    df = pd.DataFrame(output)
    filename = f"./results/{experiment}_random_query_analysis.csv"
    with open(filename, "wb") as f:
        df.to_csv(f, encoding="utf-8")
    return df


def get_table_results(df, repetition):
    our_converged_indices = np.array(
        [df["converged_w_index"].iloc[2 * k] for k in range(repetition)]
    )
    our_correct_indices = np.array(
        [df["correct_w_index"].iloc[2 * k] for k in range(repetition)]
    )
    our_questions_asked = np.array(
        [df["no_questions_asked"].iloc[2 * k] for k in range(repetition)]
    )

    our_convergence_rate = (
        sum(our_converged_indices == our_correct_indices) / our_converged_indices.shape[0]
    )
    our_mean = np.mean(our_questions_asked)
    our_std = np.std(our_questions_asked)
    our_median = np.median(our_questions_asked)

    random_converged_indices = np.array(
        [df["converged_w_index"].iloc[2 * k + 1] for k in range(repetition)]
    )
    random_correct_indices = np.array(
        [df["correct_w_index"].iloc[2 * k + 1] for k in range(repetition)]
    )
    random_questions_asked = np.array(
        [df["no_questions_asked"].iloc[2 * k + 1] for k in range(repetition)]
    )

    random_convergence_rate = (
        sum(random_converged_indices == random_correct_indices)
        / random_converged_indices.shape[0]
    )
    random_mean = np.mean(random_questions_asked)
    random_std = np.std(random_questions_asked)
    random_median = np.median(random_questions_asked)

    return [
        [our_convergence_rate, our_mean, our_std, our_median],
        [random_convergence_rate, random_mean, random_std, random_median],
    ]


def save_stats(stats):
    """
    Save and print results in Table1 of the paper.
    """
    table = pd.DataFrame(
        columns=[
            "Pedestrian Convergence Rate",
            "Overtake Convergence Rate",
            "Pedestrian Mean",
            "Overtake Mean",
            "Pedestrian Standard Deviation",
            "Overtake Standard Deviation",
            "Pedestrian Median",
            "Overtake Median",
        ]
    )

    table["Pedestrian Convergence Rate"] = [stats[0][0][0], stats[0][1][0]]
    table["Overtake Convergence Rate"] = [stats[1][0][0], stats[1][1][0]]

    table["Pedestrian Mean"] = [stats[0][0][1], stats[0][1][1]]
    table["Overtake Mean"] = [stats[1][0][1], stats[1][1][1]]

    table["Pedestrian Standard Deviation"] = [stats[0][0][2], stats[0][1][2]]
    table["Overtake Standard Deviation"] = [stats[1][0][2], stats[1][1][2]]

    table["Pedestrian Median"] = [stats[0][0][3], stats[0][1][3]]
    table["Overtake Median"] = [stats[1][0][3], stats[1][1][3]]

    print(table)
    table.to_csv("./results/SAPL_table1.csv", encoding="utf-8")


def main():
    """
    Main entry point for the script.
    Read data, initialize experiment parameters, and calls apl_experiment.
    """
    args = create_arguments()
    repeatability_evaluation = args.repeatability_evaluation

    if repeatability_evaluation:  # reproduce experiment in Table1 of the paper
        no_samples = 1000
        threshold_probability = 0.99
        repetition = 100

        stats = [[], []]
        for i, exp in enumerate(["pedestrian", "overtake"]):
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
                repetition,
                exp,
            )
            stats[i] = get_table_results(df, repetition)
        save_stats(stats)
    else:
        no_samples = args.no_samples
        threshold_probability = args.terminating_condition
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
            repetition,
            experiment,
        )


if __name__ == "__main__":
    main()
