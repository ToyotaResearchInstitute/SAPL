"""
This module includes classes related to Safe Active Preference Learning framework.

Classes:
    SAPL: Implements the Safe Active Preference Learning framework.
    fft_APL: Implements the Active Preference Learning framework using FFT.

Author: Ruya Karagulle
Date: July 2023
"""

import numpy as np
import torch
import itertools as it
from scipy.fft import fft2

import WSTL

import question_asker
import preferenceGUI
from pathlib import Path

import random

torch.set_default_dtype(torch.float64)


class SAPL:
    """
    Implements the Safe Active Preference Learning (SAPL) framework.

    Attributes:
        signals: Preprocessed signals.
        no_samples: Number of samples.
        questions: List of all possible pairs of indices for questions.
        robustness: Robustness values for the given formula and signals.
        formula: Scaled WSTL formula.
        prior_w: Prior probabilities for weight valuation.
        debug: Boolean indicating whether debugging is enabled.

    Methods:
        get_robustness_differences: Computes robustness differences for
                                    all pairs in the question list.
        scale_w_samples: Scales up root-layer weights to satisfy a lower threshold
                         for robustness differences.
        compute_bradley_terry: Computes Bradley-Terry likelihood functions for all pairs
                               in the question list.
        compute_objective: Computes the expected information gain for each pair.
        query_selection: Selects the next query based on expected information gain.
        show_trajectories: Plays trajectories on the simulator or as videos.
        w_update: Posterior update for weight valuation probabilities.

        check_agreement: Computes agreement between two realizations for
                         a given question list.
        check_user_agreement: Checks user agreement of a realization.

        convergence: Synthetic experiments to test convergence with
                     the ground truth weight valuation in the sample set.
        ood_convergence: Synthetic experiments to test convergence when
                         the ground truth is out of distribution.
        noisy_convergence: Synthetic experiments to test convergence
                           when answers are noisy.
        random_selection: Synthetic experiments to test convergence
                          when questions are selected randomly.
        user_experiment: User experiment, asking questions interactively.
    """

    def __init__(
        self,
        signals: tuple,
        formula: WSTL.WSTL_Formula,
        no_samples: int,
        robustness_difference_limit: float = 0,
        seed: int or None = None,
        debug: bool = False,
    ):
        """
        Initializes the SAPL (Safe Active Preference Learning) instance.

        Args:
            signals (tuple): Preprocessed signals.
            formula (WSTL.WSTL_Formula): WSTL formula.
            no_samples (int): Number of samples.
            robustness_difference_limit (float): Robustness difference limit.
            seed (int, optional): Seed for reproducibility. Default is None.
            debug (bool, optional): Debug mode. Default is False.
        """

        assert isinstance(
            formula, WSTL.WSTL_Formula
        ), "Formula needs to be an WSTL formula."

        assert isinstance(signals, tuple), "Signals needs to be tuple of Expressions."

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.signals = signals  # preprocessed signals
        self.no_samples = no_samples

        idx_signals = np.arange(WSTL.get_input_length(signals))
        self.questions = list(it.combinations(idx_signals, 2))

        self.robustness = formula.robustness(signals, scale=-1).squeeze(1).squeeze(-1)
        self.formula = self.scale_w_samples(formula, robustness_difference_limit)
        self.robustness = formula.robustness(signals, scale=-1).squeeze(1).squeeze(-1)

        self.prior_w = 1 / no_samples * torch.ones(no_samples)
        self.debug = debug

    def get_robustness_differences(self):
        """
        Computes robustness differences for all pairs in the question list.

        Returns:
            torch.Tensor: Matrix of robustness differences.
        """
        robustness_differences = torch.empty(len(self.questions), self.no_samples)
        for j, q in enumerate(self.questions):
            robustness_differences[j, :] = (
                self.robustness[q[0], :] - self.robustness[q[1], :]
            )
        return robustness_differences

    def scale_w_samples(
        self, formula: WSTL.WSTL_Formula, robustness_difference_limit: float
    ):
        """
        Scales up root-layer weights to satisfy lower threshold for
        robustness differences.

        Args:
            formula (WSTL.WSTL_Formula): WSTL formula.
            robustness_difference_limit (float): Robustness difference limit.

        Returns:
            WSTL.WSTL_Formula: Scaled WSTL formula.
        """
        assert isinstance(
            formula, WSTL.WSTL_Formula
        ), "Formula needs to be an WSTL formula."

        abs_rob_diffs = torch.abs(self.get_robustness_differences())
        min_abs_rob_diffs = torch.min(abs_rob_diffs, axis=0)[0]
        alphas = robustness_difference_limit / min_abs_rob_diffs

        formula.weights[list(formula.weights.keys())[-1]] = (
            alphas * formula.weights[list(formula.weights.keys())[-1]]
        )
        formula.update_weights()
        return formula

    def compute_bradley_terry(self):
        """
        Computes Bradley-Terry likelihood functions for all pairs in
        the question list.
        """

        # Folllowing nested for loop is vectorized in the function body.
        # above one-liner is equivalent to:
        # self.probability_bt[:,j,:] = ( ( (1-answers)*exps[q[0], :].unsqueeze(-1)
        #                                 + answers*exps[q[1], :].unsqueeze(-1)).T
        #                                 / (exps[q[0], :] + exps[q[1], :])).T

        self.probability_bt = torch.empty(self.no_samples, len(self.questions), 2)
        answers = torch.tensor([0, 1])

        for j, q in enumerate(self.questions):
            robustness_diff = self.robustness[q[0], :] - self.robustness[q[1], :]
            robustness_diff[robustness_diff > 709] = 709
            exp = torch.exp(robustness_diff).unsqueeze(-1)

            self.probability_bt[:, j, :] = (exp + (answers) * (1 - exp)) / (1 + exp)

    def compute_objective(self):
        """
        Computes the expected information gain for each pair.

        Returns:
            torch.Tensor: Expected information gain for each pair.
        """

        # Folllowing nested for loop isvectorized in the function body.
        # for k in range(len(self.questions)):
        #     q_cost[k] = 0
        #     for w in range(self.no_samples):
        #         for answer in [0,1]:
        #             q_cost[k] += self.p_bt[w,k,answer]*self.prior_w[w]
        #                           *torch.log2(self.p_bt[w,k,answer]/
        #                           torch.sum(self.p_bt[:, k, answer]*self.prior_w))

        self.compute_bradley_terry()
        q_cost = torch.empty(len(self.questions))
        q_cost = torch.sum(
            torch.sum(
                self.probability_bt.T
                * self.prior_w
                * torch.log2(
                    1
                    + self.probability_bt
                    / torch.sum(self.probability_bt.T * self.prior_w, axis=-1).T
                ).T,
                axis=0,
            ),
            axis=-1,
        )
        return q_cost

    def query_selection(self):
        """
        Selects the next query based on expected information gain.

        Returns:
            Tuple[Tuple[int, int], int]: Selected query and its index.
        """
        loss = self.compute_objective()
        q = torch.argmax(loss)
        return self.questions[q], q

    def show_trajectories(
        self,
        simulator: bool,
        selected_q: tuple,
        question_file: str or None = None,
    ):
        """
        Plays trajectories.

        Args:
            simulator (bool): If True, connects to the simulator (requires host name).
            selected_q (Tuple[int, int]): Selected query.
            question_file (str, optional): Path to the question file for simulator mode.

        Returns:
            int: User's answer indicating the preferred trajectory.

        """
        idx_signals = np.arange(WSTL.get_input_length(self.signals))
        qs = list(it.combinations(idx_signals, 2))
        for k in range(len(qs)):
            if qs[k] == selected_q:
                question_id = k

        if simulator:
            # replays trajectories on the simulator.
            question_asker.ask_a_question_interactive(
                Path(question_file), question_id, "10.110.21.114"
            )
        else:
            # replays trajectories as videos.
            preferenceGUI.GUI().play_question_videos(question_file, selected_q)

        answer = input("which trajectory do you prefer?")
        return int(answer)

    def w_update(self, q_index: int, answer: int):
        """
        Posterior update for weight valuation probabilities.

        Args:
            q_index (int): Index of the selected query.
            answer (int): User's answer indicating the preferred trajectory.

        Returns:
            torch.Tensor: Updated weight valuation probabilities.

        """
        next_w = (
            self.probability_bt[:, q_index, answer]
            * self.prior_w
            / torch.sum(self.probability_bt[:, q_index, answer] * self.prior_w)
        )
        return next_w

    def check_agreement(
        self, w_idx1: int, w_idx2: int, qs: list[tuple[int, int]] or None = None
    ):
        """Computes agreement between two realizations for a given question list.

        Args:
            w_idx1 (int): Index of the first weight valuation.
            w_idx2 (int): Index of the second weight valuation.
            qs (list, optional): List of question indices. Default is None.

        Returns:
            float: Agreement score between the two realizations.
        """

        if qs is None:
            idx_signals = np.arange(WSTL.get_input_length(self.signals))
            qs = list(it.combinations(idx_signals, 2))

        r_final = self.robustness[:, w_idx1]
        r_correct = self.robustness[:, w_idx2]
        correct_order = 0
        for q in qs:
            rq_final = r_final[q[0]] - r_final[q[1]]
            rq_correct = r_correct[q[0]].item() - r_correct[q[1]].item()
            if (rq_final > 0 and rq_correct > 0) or (rq_final < 0 and rq_correct < 0):
                correct_order += 1
        return correct_order / len(qs)

    def check_user_agreement(
        self, w_idx1: int, answers: list, qs: list[tuple[int, int]] or None = None
    ):
        """Checks user agreement of a realization.

        Args:
            w_idx1 (int): Index of the weight valuation.
            answers (List[int]): User's answers for a set of questions.
            qs (list, optional): List of question indices. Default is None.

        Returns:
            float: Agreement score between the user and the weight valuation.
        """

        if qs is None:
            idx_signals = np.arange(WSTL.get_input_length(self.signals))
            qs = list(it.combinations(idx_signals, 2))

        r_final = self.robustness[:, w_idx1]
        correct_order = 0
        for k, q in enumerate(qs):
            rq_final = r_final[q[0]] - r_final[q[1]]
            if (rq_final > 0 and answers[k] == 0) or (rq_final < 0 and answers[k] == 1):
                correct_order += 1
        return correct_order / len(qs)

    # --- EXPERIMENTS ---
    def convergence(self, threshold_w: float, max_questions: int, w_final: int):
        """
        Synthetic experiments to test convergence. In this setup,
        as the ground truth weight valuation is in the sample set,
        it checks the convergence to to the ground truth and returns
        statistics.

        Args:
            threshold_w (float): Convergence threshold for the maximum weight probability.
            max_questions (int): Maximum number of questions to ask.
            w_final (int): Index of the ground truth weight valuation.

        Returns:
            Dict: Convergence statistics.

        """
        max_w = torch.max(self.prior_w)
        max_w_list = [max_w]
        questions_asked = 0

        w_star_idx = 0
        q_order = []

        while (
            max_w < threshold_w
            and self.questions != []
            and questions_asked < max_questions
        ):
            selected_q, q_idx = self.query_selection()
            q_order.append(selected_q)

            self.questions.remove(selected_q)
            if (
                self.robustness[selected_q[0], w_final]
                > self.robustness[selected_q[1], w_final]
            ):
                answer = 0
            else:
                answer = 1

            posterior_w = self.w_update(q_idx, answer)

            max_w = torch.max(posterior_w)
            max_w_list.append(max_w)
            w_star_idx = torch.argmax(posterior_w)
            self.prior_w = posterior_w
            questions_asked += 1

        q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
        q_agreed = self.check_agreement(w_star_idx, w_final)

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            print(f"most-likely prob: {posterior_w[w_star_idx]}")
            print(f"converged?: {w_star_idx.item(), w_final}")
            q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            q_agreed = self.check_agreement(w_star_idx, w_final)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "converged_w_index": w_star_idx.item(),
            "correct_w_index": w_final,
            "probability": max_w.item(),
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def ood_convergence(
        self, threshold_w: float, max_questions: int, correct_robustness: torch.Tensor
    ):
        """
        Synthetic experiments to test convergence when the ground truth is
        out of distribution.

        Args:
            threshold_w (float): Convergence threshold for the maximum weight probability.
            max_questions (int): Maximum number of questions to ask.
            correct_robustness (torch.Tensor): Robustness with the correct valuation.

        Returns:
            Dict: Convergence statistics.

        """
        max_w = torch.max(self.prior_w)
        max_w_list = [max_w]
        questions_asked = 0

        w_star_idx = 0
        q_order = []

        answers = []
        while (
            max_w < threshold_w
            and self.questions != []
            and questions_asked < max_questions
        ):
            selected_q, q_idx = self.query_selection()
            q_order.append(selected_q)

            self.questions.remove(selected_q)
            if correct_robustness[selected_q[0]] > correct_robustness[selected_q[1]]:
                answer = 0
            else:
                answer = 1

            answers.append(answer)
            posterior_w = self.w_update(q_idx, answer)

            max_w = torch.max(posterior_w)
            max_w_list.append(max_w)
            w_star_idx = torch.argmax(posterior_w)
            self.prior_w = posterior_w
            questions_asked += 1

        q_agreed_over_asked = self.check_user_agreement(w_star_idx, answers, q_order)
        all_answers = []
        idx_signals = np.arange(WSTL.get_input_length(self.signals))
        qs = list(it.combinations(idx_signals, 2))
        for q in qs:
            if correct_robustness[q[0]] > correct_robustness[q[1]]:
                answer = 0
            else:
                answer = 1
            all_answers.append(answer)
        q_agreed = self.check_user_agreement(w_star_idx, all_answers)

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            print(f"most-likely prob: {posterior_w[w_star_idx]}")
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "converged_w_index": w_star_idx.item(),
            "probability": max_w.item(),
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def noisy_convergence(self, threshold_w: float, max_questions: int, w_final: int):
        """
        Synthetic experiments to test convergence when answers are noisy.

        Args:
            threshold_w (float): Convergence threshold for the maximum weight probability.
            max_questions (int): Maximum number of questions to ask.
            w_final (int): Index of the ground truth weight valuation.

        Returns:
            Dict: Convergence statistics.

        """
        max_w = torch.max(self.prior_w)
        max_w_list = [max_w]
        questions_asked = 0

        w_star_idx = 0
        q_order = []
        while (
            max_w < threshold_w
            and self.questions != []
            and questions_asked < max_questions
        ):
            selected_q, q_idx = self.query_selection()
            q_order.append(selected_q)

            noise = 0.6 + 0.4 * np.random.rand(1)
            if (
                max(
                    self.probability_bt[w_final, q_idx, 0].item(),
                    self.probability_bt[w_final, q_idx, 1].item(),
                )
                > noise
            ):
                if (
                    self.probability_bt[w_final, q_idx, 0].item()
                    > self.probability_bt[w_final, q_idx, 1].item()
                ):
                    answer = 0
                else:
                    answer = 1
            else:
                answer = int(np.random.choice(np.array([0, 1])))

            posterior_w = self.w_update(q_idx, answer)

            max_w = torch.max(posterior_w)
            max_w_list.append(max_w)
            w_star_idx = torch.argmax(posterior_w)
            self.prior_w = posterior_w
            questions_asked += 1

        q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
        q_agreed = self.check_agreement(w_star_idx, w_final)

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            print(f"most-likely prob: {posterior_w[w_star_idx]}")
            print(f"converged?: {w_star_idx.item(), w_final}")
            q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            q_agreed = self.check_agreement(w_star_idx, w_final)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "noise": noise,
            "converged_w_index": w_star_idx.item(),
            "correct_w_index": w_final,
            "probability": max_w.item(),
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def random_selection(self, threshold_w: float, w_final: int, random: bool = False):
        """
        Synthetic experiments to test convergence when questions are selected randomly.

        Args:
            threshold_w (float): Convergence threshold for the maximum weight probability.
            w_final (int): Index of the ground truth weight valuation.
            random (bool, optional): If True, questions are selected randomly.
                                     Default is False.

        Returns:
            Dict: Convergence statistics.

        """
        max_w = torch.max(self.prior_w)
        max_w_list = [max_w]
        questions_asked = 0

        w_star_idx = 0
        q_order = []
        while max_w < threshold_w and self.questions != []:
            if random:
                self.compute_bradley_terry()
                q_idx = int(torch.randint(len(self.questions), size=(1,)))
                selected_q = self.questions[q_idx]
            else:
                selected_q, q_idx = self.query_selection()
            q_order.append(selected_q)

            self.questions.remove(selected_q)
            if (
                self.robustness[selected_q[0], w_final]
                > self.robustness[selected_q[1], w_final]
            ):
                answer = 0
            else:
                answer = 1

            posterior_w = self.w_update(q_idx, answer)
            max_w = torch.max(posterior_w)
            max_w_list.append(max_w)
            w_star_idx = torch.argmax(posterior_w)
            self.prior_w = posterior_w
            questions_asked += 1

        q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
        q_agreed = self.check_agreement(w_star_idx, w_final)

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            q_agreed = self.check_agreement(w_star_idx, w_final)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "converged_w_index": w_star_idx.item(),
            "correct_w_index": w_final,
            "probability": max_w.item(),
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def user_experiment(
        self, simulator: bool, question_file: str, threshold_w: float, max_questions: int
    ):
        """
        User experiment. It asks questions interactively.

        Args:
            simulator (bool): If True, connects to the simulator (requires host name).
            question_file (str): Path to the question file for simulator mode.
            threshold_w (float): Convergence threshold for the maximum probability.
            max_questions (int): Maximum number of questions to ask.

        Returns:
            Tuple: User experiment results.

        """
        max_w = torch.max(self.prior_w)
        questions_asked = 0
        w_star_idx = 0
        q_order = []
        answers = []
        while (
            max_w < threshold_w
            and self.questions is not None
            and questions_asked < max_questions
        ):
            selected_q, q_idx = self.query_selection()
            self.questions.remove(selected_q)
            q_order.append(selected_q)

            answer = self.show_trajectories(simulator, selected_q, question_file)
            answers.append(answer)
            posterior_w = self.w_update(q_idx, answer)
            max_w = torch.max(posterior_w)
            w_star_idx = torch.argmax(posterior_w)
            self.prior_w = posterior_w
            questions_asked += 1
            if self.debug:
                print(f"Question Selected: {selected_q}")
                print(
                    f"Robustness wrt max weight set: \
                        {self.robustness[selected_q[0], w_star_idx].item()}, \
                            {self.robustness[selected_q[1], w_star_idx].item()}"
                )
                print(f"Probability of max weight: {max_w.item()}")

        agreed_answers = self.check_user_agreement(w_star_idx, answers, q_order)
        return (
            w_star_idx,
            self.robustness[:, w_star_idx],
            self.formula,
            self.questions,
            max_w,
            agreed_answers,
            questions_asked,
        )


class fft_APL:
    """
    Implements Active Preference Learning using Fast Fourier Transform (FFT).

    Attributes:
        signals: Preprocessed signals.
        no_samples: Number of samples.
        questions: List of all possible pairs of indices for questions.
        robustness: Robustness values for the given formula and signals.
        prior_w: Prior probabilities for weight valuation.
        debug: Boolean indicating whether debugging is enabled.

    Methods:
        bradley_terry_w_fft: Computes Bradley-Terry likelihood functions using
                             FFT for all pairs in the question list.
        fft_query_selection: Selects the next query based on FFT
                             expected information gain.
        compute_fft_objective: Computes the expected information gain
                               for each pair using FFT.
        w_update_fft: Posterior update for weight valuation probabilities using FFT.
        check_fft_user_agreement: Checks user agreement of a realization using FFT.
        BTfft: Active preference learning experiment using FFT.

    """

    def __init__(
        self,
        signals: tuple,
        formula: WSTL.WSTL_Formula,
        no_samples: int,
        seed: int or None = None,
        debug: bool = False,
    ):
        """
        Initializes the fft_APL (Active Preference Learning using FFT) instance.

        Args:
            signals (tuple): Preprocessed signals.
            formula (WSTL.WSTL_Formula): WSTL formula.
            no_samples (int): Number of samples.
            seed (int, optional): Seed for reproducibility. Default is None.
            debug (bool, optional): Debug mode. Default is False.
        """
        assert isinstance(
            formula, WSTL.WSTL_Formula
        ), "Formula needs to be an WSTL formula."

        assert isinstance(signals, tuple), "Signals needs to be tuple of Expressions."

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.signals = signals  # preprocessed signals
        self.no_samples = no_samples

        idx_signals = np.arange(WSTL.get_input_length(signals))
        self.questions = list(it.combinations(idx_signals, 2))

        self.robustness = formula.robustness(signals, scale=-1).squeeze(1).squeeze(-1)
        self.prior_w = 1 / no_samples * torch.ones(no_samples)
        self.debug = debug

    def bradley_terry_w_fft(self, w_set: np.array):
        """
        Takes fft of the signal and computes Bradley-Terry likelihood functions
        for all pairs in the question list.

        Args:
            w_set: Set of weights.
        """
        self.probability_bt_fft = torch.empty(self.no_samples, len(self.questions), 2)
        answers = torch.tensor([0, 1])
        fft = torch.empty((len(self.signals[0][0]), 5 * len(self.signals[0][0][0])))
        for i in range(len(self.signals[0][0])):
            signal_array = np.concatenate(
                (
                    self.signals[0][0][i],
                    self.signals[0][1][i],
                    self.signals[1][0][0][i],
                    self.signals[1][0][1][i],
                    self.signals[1][1][i],
                ),
                axis=0,
            )

            fft[i, :] = torch.sort(
                torch.tensor(np.abs(fft2(signal_array)), dtype=torch.float64).flatten()
            )[0]

        self.fft_signals = fft @ w_set

        for j, q in enumerate(self.questions):
            fft_diff = self.fft_signals[q[0], :] - self.fft_signals[q[1], :]

            fft_diff[fft_diff > 20] = 20
            fft_diff[fft_diff < -20] = -20
            exp = torch.exp(fft_diff).unsqueeze(-1)
            self.probability_bt_fft[:, j, :] = (exp + (answers) * (1 - exp)) / (1 + exp)

    def fft_query_selection(self, w_set: np.array):
        """
        Selects the next query based on FFT expected information gain.

        Args:
            w_set: Set of weights.

        Returns:
            Tuple: Selected query and its index.
        """
        loss = self.compute_fft_objective(w_set)
        q = torch.argmax(loss)

        return self.questions[q], q

    def compute_fft_objective(self, w_set: np.array):
        """
        Computes the expected information gain for each pair.

        Args:
            w_set: Set of weights.

        Returns:
            torch.Tensor: Expected information gain for each pair.
        """
        self.bradley_terry_w_fft(w_set)
        q_cost = torch.empty(len(self.questions))
        q_cost = torch.sum(
            torch.sum(
                self.probability_bt_fft.T
                * self.prior_w
                * torch.log2(
                    1
                    + self.probability_bt_fft
                    / torch.sum(self.probability_bt_fft.T * self.prior_w, axis=-1).T
                ).T,
                axis=0,
            ),
            axis=-1,
        )

        return q_cost

    def w_update_fft(self, q_index: int, answer: int):
        """Updates posteriors for weight valuation probabilities

        Args:
            q_index (int): Index of the selected query.
            answer (int): User's answer indicating the preferred trajectory.

        Returns:
            torch.Tensor: Updated weight valuation probabilities.
        """
        next_w = (
            self.probability_bt_fft[:, q_index, answer]
            * self.prior_w
            / torch.sum(self.probability_bt_fft[:, q_index, answer] * self.prior_w)
        )
        return next_w

    def check_fft_user_agreement(
        self, w_idx1: int, answers: int, qs: list or None = None
    ):
        """
        Checks user agreement of a realization.

        Args:
            w_idx1 (int): Index of the weight valuation.
            answers (List[int]): User's answers for a set of questions.
            qs (List, optional): List of question indices. Default is None.

        Returns:
            float: Agreement score between the user and the weight valuation.
        """
        if qs is None:
            idx_signals = np.arange(WSTL.get_input_length(self.signals))
            qs = list(it.combinations(idx_signals, 2))

        fft_final = self.fft_signals[:, w_idx1]
        correct_order = 0
        for k, q in enumerate(qs):
            rq_final = fft_final[q[0]] - fft_final[q[1]]
            if (rq_final > 0 and answers[k] == 0) or (rq_final < 0 and answers[k] == 1):
                correct_order += 1
        return correct_order / len(qs)

    def BTfft(
        self,
        fft_w_set: np.array,
        threshold_w: float,
        max_questions: int,
        correct_robustness: torch.Tensor,
        noise: bool = False,
    ):
        """
        Active preference learning experiment using FFT.

        Args:
            fft_w_set: Set of weights.
            threshold_w (float): Convergence threshold for the maximum weight probability.
            max_questions (int): Maximum number of questions to ask.
            correct_robustness (torch.Tensor): Robustness for the correct valuation.
            noise (bool, optional): If True, introduces noise to answers.
                                    Default is False.

        Returns:
            Tuple[int, Dict[str, Union[int, float]]]: Experiment results.
        """
        self.prior_w = 1 / fft_w_set.shape[1] * torch.ones(fft_w_set.shape[1])
        idx_signals = np.arange(WSTL.get_input_length(self.signals))
        self.questions = list(it.combinations(idx_signals, 2))

        max_w = torch.max(self.prior_w)
        max_w_list = [max_w]
        questions_asked = 0

        w_star_idx = 0
        q_order = []
        answers = []
        while (
            max_w < threshold_w
            and self.questions != []
            and questions_asked < max_questions
        ):
            selected_q, q_idx = self.fft_query_selection(fft_w_set)
            q_order.append(selected_q)

            self.questions.remove(selected_q)
            if noise is False:
                if correct_robustness[selected_q[0]] > correct_robustness[selected_q[1]]:
                    answer = 0
                else:
                    answer = 1
            else:
                if correct_robustness[selected_q[0]] > correct_robustness[selected_q[1]]:
                    answer_array = np.concatenate(
                        (
                            np.zeros((100 - int(100 * noise),)),
                            np.ones((int(100 * noise),)),
                        )
                    )
                else:
                    answer_array = np.concatenate(
                        (
                            np.ones((100 - int(100 * noise),)),
                            np.zeros((int(100 * noise),)),
                        )
                    )
                random.shuffle(answer_array)
                answer = int(random.choice(answer_array))

            posterior_w = self.w_update_fft(q_idx, answer)
            answers.append(answer)

            max_w = torch.max(posterior_w)
            max_w_list.append(max_w)
            w_star_idx = torch.argmax(posterior_w)

            self.prior_w = posterior_w
            questions_asked += 1

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            print(f"most-likely prob: {posterior_w[w_star_idx]}")
            q_agreed_over_asked = self.check_fft_user_agreement(
                w_star_idx, answers, q_order
            )
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            all_answers = []
            idx_signals = np.arange(WSTL.get_input_length(self.signals))
            qs = list(it.combinations(idx_signals, 2))
            for q in qs:
                if correct_robustness[q[0]] > correct_robustness[q[1]]:
                    answer = 0
                else:
                    answer = 1
                all_answers.append(answer)
            q_agreed = self.check_fft_user_agreement(w_star_idx, all_answers)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return w_star_idx, {
            "noise": noise,
            "converged_w_index": w_star_idx.item(),
            "probability": max_w,
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }
