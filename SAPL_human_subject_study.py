"""
This program runs human subject studies on either a computer or a simulator
on two scenarios: pedestrian and overtake.

It takes experiment conditions: number of signals to create questions from,
                                number of samples to select among,
                                terminating condition,
                                number of questions to ask,
                                number of validation questions,
                                experiment type,
                                state of the simulator (exists or not)

and returns a .csv file (or adds a row to an existing .csv file)
that contains: most-likely weight valuation
               list of weighted robustness values of signals w/ most-likely valuation
               posterior probability of the most-likely weight valuation
               number of training questions
               user agreement in training questions
               user agreement in validation questions.

Author: Ruya Karagulle
Date: July 2023
"""

import os
import sys
import pickle

import numpy as np
import argparse
import pandas as pd
import torch

cwd = os.getcwd()
sys.path.insert(0, f"{cwd}/src")

from APL_utils import *  # NOQA
from preprocess_utils import *  # NOQA


def create_arguments():
    """function to parse arguments"""
    parser = argparse.ArgumentParser(
        description="The program conducts human subject studies."
    )
    parser.add_argument(
        "--no_signals",
        default=16,
        type=int,
        help="Number of traces to create questions to the participant",
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
        default=12,
        type=int,
        help=" Number of questions to be asked to the participant.",
    )
    parser.add_argument(
        "--validation_questions",
        default=0,
        type=int,
        help=" Number of validation questions to be asked to the participant.",
    )
    parser.add_argument(
        "--experiment",
        default="pedestrian",
        type=str,
        help="Experiment type. Options: pedestrian, overtake ",
    )
    parser.add_argument(
        "--simulator",
        default=False,
        type=bool,
        help="True if motion base simulator exists, false otherwise.",
    )
    return parser.parse_args()


def aPL_experiment(
    signals: tuple,
    formula: WSTL.WSTL_Formula,
    no_samples: int,
    threshold_probability: float,
    no_questions: int,
    validation: int,
    filename: str,
    question_file: str,
    simulator: bool,
):
    """
    Takes experiment variables, runs the experiment, saves results to .csv.

    Args:
        signals (tuple): set of trajectories to be used in the experiment.
        formula (WSTL_Formula): formula candidate used to assess robustness values
        no_samples (int): number of weight valuation samples
        threshold_probability (float): threshold posterior probailibty for termination
        no_questions (int): maximum number of questions to be asked
        validation (int): number of validation questions
        filename (str): file to save the results
        question_file (str): replayer data for questions
        simulator (bool): status of the motion base simulator.

    Returns:
        A .csv file with statistics on the experiment
    """

    # set the weight valuation sample set
    formula.set_weights(signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True)

    # set robustness difference bound
    u = 0.36
    rob_diff_bound = -np.log(u / (1 - u))

    # set active learning instance
    aPL_instance = SAPL(
        signals,
        formula=formula,
        no_samples=no_samples,
        robustness_difference_limit=rob_diff_bound,
        debug=True,
    )

    # Training
    outputs = aPL_instance.user_experiment(
        simulator, question_file, threshold_probability, no_questions
    )
    [
        most_likely_w_set,
        decided_robustness,
        formula,
        remaining_questions,
        max_w,
        agreed_answers,
        no_questions_asked,
    ] = outputs

    # Validation
    aligned_validation_questions = 0
    if validation != 0:
        for _ in range(validation):
            q_idx = int(torch.randint(len(remaining_questions), size=(1,)))
            selected_q = remaining_questions[q_idx]
            remaining_questions.remove(selected_q)

            answer = aPL_instance.show_trajectories(selected_q)
            r_diff = decided_robustness[selected_q[0]] - decided_robustness[selected_q[1]]
            if (r_diff > 0 and answer == 0) or (r_diff < 0 and answer == 1):
                aligned_validation_questions += 1
        validation_agreement = aligned_validation_questions / validation
    else:
        validation_agreement = False

    final_weightset = get_final_w_set(formula, most_likely_w_set)  # NOQA

    output = {
        "final_w_set": [final_weightset],
        "decided_robustness": [decided_robustness.detach().numpy()],
        "posterior of the most likely weight set": max_w,
        "no_train_questions": no_questions_asked,
        "train_data_aligned_answers": agreed_answers,
        "validation_aligned_answers": validation_agreement,
    }

    df = pd.DataFrame(output)
    df.to_csv(filename, encoding="utf-8", mode="a")


def main():
    """
    Enters the script.
    Sets arguments and complete preprocess, then run the experiment.

    Returns:
        A .csv file with statistics on the experiment
    """
    args = create_arguments()

    no_samples = args.no_samples
    threshold_probability = args.terminating_condition
    no_questions = args.no_questions
    experiment = args.experiment
    validation = args.validation_questions
    simulator = args.simulator

    # read trajectory data
    data_name = f"./data/{args.experiment}_trajectories.pkl"
    with open(data_name, "rb") as f:
        data = pickle.load(f)

    # setup experiment variables
    processed_signals = get_signals(data, experiment)
    phi = get_formula(processed_signals, experiment)

    filename = f"./results/SAPL_HSS_{experiment}.csv"
    if simulator:
        question_file = f"./data/{experiment}_question.csv"
    else:
        question_file = f"/Users/rkaragulle/Documents/replayer_data/{experiment}"

    aPL_experiment(
        processed_signals,
        phi,
        no_samples,
        threshold_probability,
        no_questions,
        validation,
        filename,
        question_file,
        simulator,
    )


if __name__ == "__main__":
    main()
