"""
This codebase includes functions related to preference learning framework.

author: Ruya Karagulle
date: July 2023
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
    def __init__(
        self,
        signals,
        formula,
        no_samples,
        robustness_difference_limit=0,
        seed=None,
        debug=False,
    ):

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
        this function computes robustness differences for all pairs in the question list
        """
        robustness_differences = torch.empty(len(self.questions), self.no_samples)
        for j, q in enumerate(self.questions):
            robustness_differences[j, :] = (
                self.robustness[q[0], :] - self.robustness[q[1], :]
            )
        return robustness_differences

    def scale_w_samples(self, formula, robustness_difference_limit):
        """
        this function scales up root-layer weights to satisfy lower threshold for
        robustness differences
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
        this function computes Bradley-Terry likelihood functions for all pairs in
        the question list.
        """
        self.probability_bt = torch.empty(self.no_samples, len(self.questions), 2)
        answers = torch.tensor([0, 1])

        for j, q in enumerate(self.questions):
            robustness_diff = self.robustness[q[0], :] - self.robustness[q[1], :]
            robustness_diff[robustness_diff > 709] = 709
            exp = torch.exp(robustness_diff).unsqueeze(-1)

            self.probability_bt[:, j, :] = (exp + (answers) * (1 - exp)) / (1 + exp)

            # above one-liner is equivalent to:
            # self.probability_bt[:,j,:] = ( ( (1-answers)*exps[q[0], :].unsqueeze(-1)
            #                                + answers*exps[q[1], :].unsqueeze(-1)).T
            #                                / (exps[q[0], :] + exps[q[1], :])).T

    def compute_objective(self):
        """
        This computes the expected information gain for each pair.

        For future sanity checks, the folllowing nested for loop is
        vectorized in this code.
        for k in range(len(self.questions)):
            q_cost[k] = 0
            for w in range(self.no_samples):
                for answer in [0,1]:
                    q_cost[k] += self.p_bt[w,k,answer]*self.prior_w[w]
                                  *torch.log2(self.p_bt[w,k,answer]/
                                  torch.sum(self.p_bt[:, k, answer]*self.prior_w))
        """
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
        this function selects the next query based on expected information gain
        """
        loss = self.compute_objective()
        q = torch.argmax(loss)
        return self.questions[q], q

    def show_trajectories(self, simulator, selected_q, question_file=None):
        """
        thsi function plays trajectories.
        if the simulator is set True, it connects it.
        else it plays trajectories as videos.
        it also asks users their answer.
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
            preferenceGUI.GUI().play_question_videos(selected_q)

        answer = input("which trajectory do you prefer?")
        return int(answer)

    def w_update(self, q_index, answer):
        """
        posterior update for weight valuation probabilities
        """
        next_w = (
            self.probability_bt[:, q_index, answer]
            * self.prior_w
            / torch.sum(self.probability_bt[:, q_index, answer] * self.prior_w)
        )
        return next_w

    def check_agreement(self, w_idx1, w_idx2, qs=None):
        """
        ths function computes agreement between two realization for a given question list.
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

    def check_user_agreement(self, w_idx1, answers, qs=None):
        """
        this function checks user agreement of a realization.
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
    def convergence(self, threshold_w, max_questions, w_final):
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
            "probability": max_w,
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def ood_convergence(self, threshold_w, max_questions, correct_robustness):
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

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            print(f"most-likely prob: {posterior_w[w_star_idx]}")
            q_agreed_over_asked = self.check_user_agreement(w_star_idx, answers, q_order)
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
            q_agreed = self.check_user_agreement(w_star_idx, all_answers)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "converged_w_index": w_star_idx.item(),
            "probability": max_w,
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def noisy_convergence(self, threshold_w, max_questions, w_final):
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
            "probability": max_w,
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def random_selection(self, threshold_w, w_final, random=False):
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

        if self.debug:
            print(f"number of questions asked: {questions_asked}")
            q_agreed_over_asked = self.check_agreement(w_star_idx, w_final, q_order)
            print(f"number of agreed answers on asked questions: {q_agreed_over_asked}")
            q_agreed = self.check_agreement(w_star_idx, w_final)
            print(f"number of agreed answers for all questions: {q_agreed}")

        return {
            "converged_w_index": w_star_idx.item(),
            "correct_w_index": w_final,
            "probability": max_w,
            "no_questions_asked": questions_asked,
            "no_questions_agreed_of_asked": q_agreed_over_asked,
            "no_questions_agreed": q_agreed,
        }

    def user_experiment(self, simulator, question_file, threshold_w, max_questions):
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
    def __init__(self, signals, formula, no_samples, seed=None, debug=False):
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

    def bradley_terry_w_fft(self, w_set):
        """
        this function computes Bradley-Terry likelihood functions for all pairs in
        the question list.
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

    def fft_query_selection(self, w_set):
        loss = self.compute_fft_objective(w_set)
        q = torch.argmax(loss)

        return self.questions[q], q

    def compute_fft_objective(self, w_set):
        """
        This computes the expected information gain for each pair.

        For future sanity checks, the folllowing nested for loop
        is vectorized in this code.
        for k in range(len(self.questions)):
            q_cost[k] = 0
            for w in range(self.no_samples):
                for answer in [0,1]:
                    q_cost[k] += self.p_bt[w,k,answer]*self.prior_w[w]
                                  *torch.log2(self.p_bt[w,k,answer]/
                                  torch.sum(self.p_bt[:, k, answer]*self.prior_w))
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

    def w_update_fft(self, q_index, answer):
        """
        posterior update for weight valuation probabilities
        """
        next_w = (
            self.probability_bt_fft[:, q_index, answer]
            * self.prior_w
            / torch.sum(self.probability_bt_fft[:, q_index, answer] * self.prior_w)
        )
        return next_w

    def check_fft_user_agreement(self, w_idx1, answers, qs=None):
        """
        this function checks user agreement of a realization.
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
        self, fft_w_set, threshold_w, max_questions, correct_robustness, noise=False
    ):
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
