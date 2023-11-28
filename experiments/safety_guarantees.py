"""
This program runs safety comparison experiment.

"""
import os
import sys
import pickle

import numpy as np
import random
import torch
import argparse
import pandas as pd
from scipy.fft import fft2

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

# codebase specific libraries
from APL_utils import *  # NOQA
from preprocess_utils import *  # NOQA


def create_arguments():
    """
    Defines arguments for the experiment.
    """
    parser = argparse.ArgumentParser(
        description="The program tests the safety properties"
    )
    parser.add_argument(
        "--no_samples",
        default=1000,
        type=int,
        help="Nummber of samples for weight valuations",
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
    return parser.parse_args()


def aPL_experiment(
    signals,
    formula,
    no_samples,
    threshold_probability,
    no_questions,
    repetition,
    experiment,
    viol_data,
):
    """
    Experiment setup
    """
    # set bounds
    u = 0.4
    rob_diff_bound = -np.log(u / (1 - u))

    output = []
    for i in range(repetition):
        np.random.seed(i)

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

        # random weights for fft setup
        fft_w_set = -0.01 + 0.01 * np.random.rand(5 * len(signals[0][0][0]), no_samples)

        # experiment variables setup
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
        fft_instance = fft_APL(
            signals, formula=formula, no_samples=no_samples, seed=i, debug=True
        )
        w_star, res = fft_instance.BTfft(
            fft_w_set, threshold_probability, no_questions, final_robs
        )
        output.append(res)
        counter = 0
        for i in range(len(signals[0][0])):
            signal_array = np.concatenate(
                (
                    signals[0][0][i],
                    signals[0][1][i],
                    signals[1][0][0][i],
                    signals[1][0][1][i],
                    signals[1][1][i],
                ),
                axis=0,
            )
            viol_array = np.concatenate(
                (
                    viol_data[0][0],
                    viol_data[0][1],
                    viol_data[1][0][0],
                    viol_data[1][0][1],
                    viol_data[1][1],
                ),
                axis=0,
            )
            # print(signal_array.shape)
            fft_sig = torch.sort(
                torch.tensor(np.abs(fft2(signal_array)), dtype=torch.float64).flatten()
            )[0]
            fft_viol = torch.sort(
                torch.tensor(np.abs(fft2(viol_array)), dtype=torch.float64).flatten()
            )[0]

            if (fft_sig - fft_viol) @ fft_w_set[:, w_star] < 0:
                counter += 1
        print(counter)

    df = pd.DataFrame(output)
    df.to_csv(f"./results/{experiment}_fft_compare_unlimited_q.csv", encoding="utf-8")


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

    if experiment == "pedestrian":
        print("No violating daytsa exists. Try overtake scenario")
        return

    zero_rob_data = {"ego_trajectory": [], "ado_trajectory": []}
    zero_rob_data["ego_trajectory"].append(data["ego_trajectory"][10])
    zero_rob_data["ado_trajectory"].append(data["ego_trajectory"][10])

    data = get_pruned_data(data, experiment)
    processed_signals = get_signals(data, experiment)
    zero_rob_processed_signal = get_signals(zero_rob_data, experiment, max_length=178)
    phi = get_formula(processed_signals, experiment)

    aPL_experiment(
        processed_signals,
        phi,
        no_samples,
        threshold_probability,
        no_questions,
        repetition,
        experiment,
        zero_rob_processed_signal,
    )


if __name__ == "__main__":
    main()
