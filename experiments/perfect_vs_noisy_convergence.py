"""


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
        "--noise_level",
        default=False,
        type=int,
        help="set conditions on w valuation uniqueness.",
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
        w_final = random.randint(0, no_samples)
        formula.set_weights(
            signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True
        )

        aPL_instance = SAPL(
            signals,
            formula=formula,
            no_samples=no_samples,
            robustness_difference_limit=rob_diff_bound,
            debug=True,
        )

        output.append(
            aPL_instance.noisy_convergence(threshold_probability, no_questions, w_final)
        )

    df = pd.DataFrame(output)
    df.to_csv(f"./results/{experiment}_noisy_convergence_analysis.csv", encoding="utf-8")


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
