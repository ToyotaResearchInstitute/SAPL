#!/usr/bin/python3

# Copyright 2023 Toyota Research Institute.  All rights reserved.
"""
Safe Active Preference Learning (APL) Experiment Script

This script conducts experiments to test the convergence properties of
the Safe Active Preference Learning (SAPL) framework.
It uses trajectories and robustness functions to analyze the
behavior of the framework in various scenarios.

This experimentis in Section VII.A.(d) and titled
"Perfomance analysis when the correct valuation is out of sample set"

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
    $ python3 convergence_analysis_ood.py
    --no_samples 1000 --terminating_condition 0.99 --experiment overtake --repetition 100

Author: Ruya Karagulle
Date: September 2023
"""
import os
import sys
import pickle
import argparse

import random
import numpy as np
import pandas as pd
import torch

# Additional dependencies
cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

import WSTL  # NOQA
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

        # create no_samples + 1 samples and choose 1 as the ground truth
        formula.set_weights(
            signals, w_range=[0.1, 1.1], no_samples=no_samples + 1, random=True, seed=i
        )
        w_final = random.randint(0, no_samples)

        robs = formula.robustness(signals, scale=-1).squeeze(1).squeeze(-1)
        final_robs = robs[:, w_final]

        # remove the ground truth from sample set
        for key in formula.weights.keys():
            formula.weights[key] = torch.cat(
                (
                    formula.weights[key][:, :w_final],
                    formula.weights[key][:, w_final + 1 :],
                ),
                axis=1,
            )
        formula.update_weights()

        apl_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
        )

        output.append(
            apl_instance.ood_convergence(threshold_probability, no_questions, final_robs)
        )

    df = pd.DataFrame(output)
    df.to_csv(
        f"./results/{experiment}_ood_convergence_analysis.csv",
        encoding="utf-8",
    )
    return df


def save_stats(dfs, repetition):
    df = pd.concat(dfs)
    pedestrian_no_questions = np.array(
        [df["no_questions_asked"].iloc[k] for k in range(repetition)]
    )
    pedestrian_train_agreement = 100 * np.array(
        [df["no_questions_agreed_of_asked"].iloc[k] for k in range(repetition)]
    )
    pedestrian_all_agreement = 100 * np.array(
        [df["no_questions_agreed"].iloc[k] for k in range(repetition)]
    )

    overtake_no_questions = np.array(
        [df["no_questions_asked"].iloc[k + repetition] for k in range(repetition)]
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

    table = pd.DataFrame()
    table["Scenario"] = ["Pedestrian", "Overtake"]
    table["Average Questions"] = [
        np.mean(pedestrian_no_questions),
        np.mean(overtake_no_questions),
    ]
    table["Mean of Train Agreement"] = [
        np.mean(pedestrian_train_agreement),
        np.mean(overtake_train_agreement),
    ]
    table["Standard Deviation of Train Agreement"] = [
        np.std(pedestrian_train_agreement),
        np.std(overtake_train_agreement),
    ]
    table["Mean of Overall Agreement"] = [
        np.mean(pedestrian_all_agreement),
        np.mean(overtake_all_agreement),
    ]
    table["Standard Deviation of Overall Agreement"] = [
        np.std(pedestrian_all_agreement),
        np.std(overtake_all_agreement),
    ]

    print(table)
    table.to_csv("./results/SAPL_table3.csv", encoding="utf-8")


def main():
    """
    Main entry point for the script.
    Read data, initialize experiment parameters, and call aPL_experiment.
    """
    args = create_arguments()
    no_questions = 20
    repeatability_evaluation = args.repeatability_evaluation
    if repeatability_evaluation:  # reproduce the experiment in Table3 of the paper
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
