"""
Safe Active Preference Learning (APL) Experiment Script


This program compares the performance of the query selection method
with random query selection.
It uses trajectories and robustness functions to analyze the
behavior of the framework in various scenarios.

This experimentis in Section VII.A.(a) and titled "The query selection performance"

The experiment results are saved in a CSV file named
'{experiment}_ood_convergence_analysis_unlimited_q.csv'.

Command-line Arguments:
    - --no_samples: Number of samples for weight sets (default: 1000).
    - --terminating_condition: Probability limit to end the learning framework
                               (default: 0.99).
    - --experiment: Type of experiment
                    (Options: 'pedestrian', 'overtake'; default: 'overtake').
    - --repetition: Number of times the test will be repeated (default: 100).

Example:
    $ python3 quesry_selection_vs_random_ask_comparison.py
    --no_samples 1000 --terminating_condition 0.99 --experiment overtake --repetition 100

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
    return parser.parse_args()


def aPL_experiment(
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
            signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True
        )

        w_final = random.randint(0, no_samples)
        aPL_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
            debug=True,
        )

        output.append(aPL_instance.random_selection(threshold_probability, w_final))
        random_apl = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
            debug=True,
        )
        output.append(
            random_apl.random_selection(threshold_probability, w_final, random=True)
        )

    df = pd.DataFrame(output)
    df.to_csv(f"./results/{experiment}_random_query_analysis.csv", encoding="utf-8")


def main():
    """
    Main entry point for the script.
    Reads data, initializes experiment parameters, and calls aPL_experiment.
    """

    args = create_arguments()
    no_samples = args.no_samples
    threshold_probability = args.terminating_condition
    experiment = args.experiment
    repetition = args.repetition

    data_name = f"./data/{args.experiment}_trajectories.pkl"
    with open(data_name, "rb") as f:
        data = pickle.load(f)

    data = get_pruned_data(data, experiment)
    processed_signals = get_signals(data, experiment)
    phi = get_formula(processed_signals, experiment)

    aPL_experiment(
        processed_signals,
        phi,
        no_samples,
        threshold_probability,
        repetition,
        experiment,
    )


if __name__ == "__main__":
    main()
