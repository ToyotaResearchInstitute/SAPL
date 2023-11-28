"""


"""
import os
import sys
import pickle

import numpy as np
import argparse
import pandas as pd
import torch
import random

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

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

    u = 0.4
    rob_diff_bound = -np.log(u / (1 - u))

    output = []
    for i in range(repetition):
        random.seed(i)

        # create no_samples + 1 samples and choose 1 as the ground truth
        formula.set_weights(
            signals, w_range=[0.1, 1.1], no_samples=no_samples + 1, random=True, seed=i
        )
        w_final = random.randint(0, no_samples + 1)

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

        aPL_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            seed=i,
            debug=True,
        )

        output.append(
            aPL_instance.ood_convergence(threshold_probability, no_questions, final_robs)
        )

    df = pd.DataFrame(output)
    df.to_csv(
        f"./results/{experiment}_ood_convergence_analysis_unlimited_q.csv",
        encoding="utf-8",
    )


def main():
    args = create_arguments()
    no_questions = 200

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
        no_questions,
        repetition,
        experiment,
    )


if __name__ == "__main__":
    main()
