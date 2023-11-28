"""


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
    parser = argparse.ArgumentParser(
        description="The program tests the convergence properties"
    )
    parser.add_argument(
        "--no_signals", default=16, type=int, help="Number of traces to be asked to user"
    )
    parser.add_argument(
        "--no_samples", default=1000, type=int, help="Nummber of samples for weight sets"
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
        help=" Number of questions to be asked to user.",
    )
    parser.add_argument(
        "--validation_questions",
        default=0,
        type=int,
        help=" Number of validation questions to be asked to user.",
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
        help="True if motion base simulator exists.",
    )
    return parser.parse_args()


def aPL_experiment(
    signals,
    formula,
    no_samples,
    threshold_probability,
    no_questions,
    validation,
    filename,
    question_file,
    simulator,
    experiment,
):

    u = 0.36
    rob_diff_bound = -np.log(u / (1 - u))

    aPL_instance = SAPL(
        signals,
        formula=formula,
        no_samples=no_samples,
        robustness_difference_limit=rob_diff_bound,
        debug=True,
    )

    if experiment == "overtake":
        pruned_questions = []
        for q in aPL_instance.questions:
            if (q[0] not in [3, 4, 10, 14, 19]) and (q[1] not in [3, 4, 10, 14, 19]):
                pruned_questions.append(q)
        aPL_instance.questions = pruned_questions

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
    args = create_arguments()

    no_samples = args.no_samples
    threshold_probability = args.terminating_condition
    no_questions = args.no_questions
    experiment = args.experiment
    validation = args.validation_questions
    simulator = args.simulator

    data_name = f"./data/{args.experiment}_trajectories.pkl"
    with open(data_name, "rb") as f:
        data = pickle.load(f)

    processed_signals = get_signals(data, experiment)
    phi = get_formula(processed_signals, experiment)

    phi.set_weights(
        processed_signals, w_range=[0.1, 1.1], no_samples=no_samples, random=True
    )

    filename = f"./results/SAPL_HSS_{experiment}.csv"
    if experiment == "pedestrian":
        question_file = f"./data/{experiment}_question.csv"
    elif experiment == "overtake":
        question_file = f"./data/{experiment}_question_filtered.csv"

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
        experiment,
    )


if __name__ == "__main__":
    main()
