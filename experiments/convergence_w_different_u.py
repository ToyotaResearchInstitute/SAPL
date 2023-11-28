"""


"""
import random
import os
import sys
import pickle

import numpy as np
import argparse
import pandas as pd

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

# codebase specific libraries
from APL_utils import *  # NOQA
from preprocess_utils import *  # NOQA


def create_arguments():
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


def aPL_experiment(
    signals,
    formula,
    no_samples,
    threshold_probability,
    no_questions,
    repetition,
    experiment,
):
    output = []

    w_final = random.randint(0, no_samples)
    formula.set_weights(signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True)

    for i in range(repetition):
        random.seed(i)

        u = (i + 1) * 0.5 / repetition
        rob_diff_bound = -np.log(u / (1 - u))
        aPL_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
            debug=True,
        )
        output.append(
            aPL_instance.convergence(threshold_probability, no_questions, w_final)
        )

    df = pd.DataFrame(output)
    df.to_csv(
        f"./results/{experiment}_different_u_convergence_analysiscsv", encoding="utf-8"
    )


def main():
    args = create_arguments()
    no_samples = args.no_samples
    threshold_probability = args.terminating_condition
    no_questions = args.no_questions
    experiment = args.experiment
    repetition = args.repetition

    data_name = f"./data/{args.experiment}_trajectories.pkl"
    with open(data_name, "rb") as f:
        data = pickle.load(f)

    if experiment == "overtake":
        data_pruned = {"ego_trajectory": [], "ado_trajectory": []}
        for k in range(len(data)):
            if k not in [3, 4, 10, 14, 19]:
                data_pruned["ego_trajectory"].append(data["ego_trajectory"][k])
                data_pruned["ado_trajectory"].append(data["ado_trajectory"][k])
        data = data_pruned

    data = get_pruned_data(data, experiment)
    processed_signals = get_signals(data, experiment)
    phi = get_formula(processed_signals, experiment)

    aPL_experiment(
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
