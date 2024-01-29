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
    formula.set_weights(signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True)

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


def main():
    args = create_arguments()

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
