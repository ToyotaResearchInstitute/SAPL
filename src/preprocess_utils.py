# Copyright 2023 Toyota Research Institute.  All rights reserved.
"""
Utils functions for preprocess signals and formulas.
These are scenario dependent functions and only defined for
scenarios in the paper.

Import these functions and use them to prepare signals
and create WSTL formulas for specific scenarios.

Author: Ruya Karagulle
Date: May 2023
"""

import numpy as np

from APL_utils import *
import WSTL
from WSTL import Expression
import pandas as pd


def get_formula(processed_signals, experiment):
    """
    Returns a WSTL formula of representing scenario specifications.
     Args:
        processed_signals (tuple): Tuple of signals for WSTL_formula computation.
        experiment (str): The type of experiment.

    Returns:
        WSTL_formula: The formulated WSTL scenario.
    """
    if experiment == "pedestrian_cphs":
        distance = Expression("distance", processed_signals[0][0])
        acceleration = Expression("acceleration", processed_signals[0][1][0])
        jerk = Expression("jerk", processed_signals[0][1][1])
        position = Expression("position", processed_signals[1][0])

        phi1 = WSTL.Always(distance >= 2)

        phi_comfort = WSTL.And(
            subformula1=WSTL.Always(acceleration <= 10),
            subformula2=WSTL.Always(jerk <= 30),
        )

        phi_destination = WSTL.Eventually(
            WSTL.Always(
                subformula=WSTL.And(
                    subformula1=position >= -271, subformula2=position <= -200
                ),
                interval=[0, 10],
            )
        )

        phi = WSTL.And(
            subformula1=WSTL.And(subformula1=phi1, subformula2=phi_comfort),
            subformula2=phi_destination,
        )

    if experiment == "pedestrian":
        # speed = Expression( 'speed' ,processed_signals[1])
        distance = Expression("distance", processed_signals[0])
        acceleration = Expression("acceleration", processed_signals[1][0])
        jerk = Expression("jerk", processed_signals[1][1])

        phi1 = WSTL.Always(distance >= 2)
        phi_comfort = WSTL.And(
            subformula1=WSTL.Always(acceleration <= 10),
            subformula2=WSTL.Always(jerk <= 30),
        )

        phi = WSTL.And(subformula1=phi1, subformula2=phi_comfort)

    elif experiment == "overtake":
        speed = Expression("speed", processed_signals[0][1])
        distance = Expression("distance", processed_signals[0][0])

        lateral_distance = Expression("lateral dist", processed_signals[1][0][0])
        longitudinal_distance = Expression(
            "longitudinal dist", processed_signals[1][0][1]
        )
        relative_velocity = Expression("relative velocity", processed_signals[1][1])

        phi1 = WSTL.Always(distance >= 2)
        phi2 = WSTL.Always(speed >= 0)

        phi_trivial = WSTL.And(
            subformula1=WSTL.And(
                subformula1=WSTL.Always(lateral_distance >= 0),
                subformula2=WSTL.Always(longitudinal_distance >= 0),
            ),
            subformula2=WSTL.Always(relative_velocity >= 0),
        )
        phi = WSTL.And(
            subformula1=WSTL.And(subformula1=phi1, subformula2=phi2),
            subformula2=phi_trivial,
        )
    return phi


def get_signals(data, experiment, N=None, max_length=None):
    """
    Prepares a tuple of signals suitable for WSTL formula robustness computation.

    Args:
        data (list): Raw signal data.
        experiment (str): Name of the scenario.
        N (int, optional): Number of signals to set.
        max_length (int, optional): Maximum length of signals.

    Returns:
        tuple: Tuple of signals formatted for WSTL formula computation.
    """

    ego_data = []
    ado_data = []
    for k in range(len(data["ego_trajectory"])):
        try:
            ego_data.append(torch.tensor(np.array(data["ego_trajectory"][k]))[:, 1:3])
            ado_data.append(torch.tensor(np.array(data["ado_trajectory"][k]))[:, 1:3])
        except KeyError:
            continue

    if N is None:
        N = len(ego_data)

    filt = (
        1 / (15) * torch.ones((1, 1, 15))
    )  # filtered instantaneous changes in CARLA speed data
    if max_length is None:
        max_length = max([len(ego_data[k]) for k in range(N)])

    acceleration = torch.ones(size=(N, max_length, 1))
    jerk = torch.ones(size=(N, max_length, 1))
    distance = torch.ones(size=(N, max_length, 1))
    position = torch.ones(size=(N, max_length, 1))

    lateral = torch.ones(size=(N, max_length, 1))
    longitudinal = torch.ones(size=(N, max_length, 1))
    relative_speed = torch.ones(size=(N, max_length, 1))
    speed_vector = torch.ones(size=(N, max_length, 1))

    for k in range(N):
        velocity = (ego_data[k][:-1, :] - ego_data[k][1:, :]) / 0.1
        speed = torch.linalg.norm(velocity, axis=-1, keepdim=True)

        if experiment == "pedestrian_cphs":
            pos = ego_data[k][:, 0].unsqueeze(0).unsqueeze(-1)
            speed = speed.unsqueeze(0).unsqueeze(0).squeeze(-1)
            speed_smooth = (
                torch.nn.functional.conv1d(speed, filt).squeeze(0).unsqueeze(-1)
            )
            accel = torch.linalg.norm(
                (speed_smooth[:, :-1, :] - speed_smooth[:, 1:, :]) / 0.1,
                axis=-1,
                keepdim=True,
            )
            jrk = torch.linalg.norm(
                (accel[:, :-1, :] - accel[:, 1:, :]) / 0.1, axis=-1, keepdim=True
            )

            dist = torch.linalg.norm(
                ego_data[k] - ado_data[k], axis=-1, keepdim=True
            ).unsqueeze(0)

            distance[k, :, :] = torch.cat(
                (
                    dist,
                    (dist[:, -1, :])
                    * torch.ones(size=(1, max_length - dist.shape[1], 1)),
                ),
                axis=1,
            )
            acceleration[k, :, :] = torch.cat(
                (
                    accel,
                    (accel[:, -1, :])
                    * torch.ones(size=(1, max_length - accel.shape[1], 1)),
                ),
                axis=1,
            )
            jerk[k, :, :] = torch.cat(
                (
                    jrk,
                    (jrk[:, -1, :]) * torch.ones(size=(1, max_length - jrk.shape[1], 1)),
                ),
                axis=1,
            )

            position[k, :, :] = torch.cat(
                (
                    pos,
                    (pos[:, -1, :]) * torch.ones(size=(1, max_length - pos.shape[1], 1)),
                ),
                axis=1,
            )
        if experiment == "pedestrian":
            speed = speed.unsqueeze(0).unsqueeze(0).squeeze(-1)
            speed_smooth = (
                torch.nn.functional.conv1d(speed, filt).squeeze(0).unsqueeze(-1)
            )
            accel = torch.linalg.norm(
                (speed_smooth[:, :-1, :] - speed_smooth[:, 1:, :]) / 0.1,
                axis=-1,
                keepdim=True,
            )
            jrk = torch.linalg.norm(
                (accel[:, :-1, :] - accel[:, 1:, :]) / 0.1, axis=-1, keepdim=True
            )

            dist = torch.linalg.norm(
                ego_data[k] - ado_data[k], axis=-1, keepdim=True
            ).unsqueeze(0)

            distance[k, :, :] = torch.cat(
                (
                    dist,
                    (dist[:, -1, :])
                    * torch.ones(size=(1, max_length - dist.shape[1], 1)),
                ),
                axis=1,
            )
            acceleration[k, :, :] = torch.cat(
                (
                    accel,
                    (accel[:, -1, :])
                    * torch.ones(size=(1, max_length - accel.shape[1], 1)),
                ),
                axis=1,
            )
            jerk[k, :, :] = torch.cat(
                (
                    jrk,
                    (jrk[:, -1, :]) * torch.ones(size=(1, max_length - jrk.shape[1], 1)),
                ),
                axis=1,
            )

        if experiment == "overtake":
            dist = torch.linalg.norm(
                ego_data[k] - ado_data[k], axis=-1, keepdim=True
            ).unsqueeze(0)

            distance[k, :, :] = torch.cat(
                (
                    dist,
                    (dist[:, -1, :])
                    * torch.ones(size=(1, max_length - dist.shape[1], 1)),
                ),
                axis=1,
            )

            speed = speed.unsqueeze(0)
            lead2ego = ado_data[k] - ego_data[k]

            #
            yaws = torch.atan2(
                (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 1],
                (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 0],
            )
            e_lateral = torch.cat(
                (torch.cos(yaws).unsqueeze(-1), torch.sin(yaws).unsqueeze(-1)), axis=-1
            )
            e_longitudinal = torch.cat(
                (-torch.sin(yaws).unsqueeze(-1), torch.cos(yaws).unsqueeze(-1)), axis=-1
            )

            long = (
                torch.abs(torch.einsum("ik,ik->i", lead2ego[:-1, :], e_longitudinal))
                .unsqueeze(-1)
                .unsqueeze(0)
            )
            lat = (
                torch.abs(torch.einsum("ik,ik->i", lead2ego[:-1, :], e_lateral))
                .unsqueeze(-1)
                .unsqueeze(0)
            )

            # ego_velocity = (ego_data[:,:-1,:]-ego_data[:,1:,:])/0.01
            ado_velocity = (ado_data[k][:-1, :] - ado_data[k][1:, :]) / 0.1

            relative_s = (
                torch.linalg.norm(velocity - ado_velocity, axis=-1)
                .unsqueeze(-1)
                .unsqueeze(0)
            )

            longitudinal[k, :, :] = torch.cat(
                (
                    long,
                    (long[:, -1, :])
                    * torch.ones(size=(1, max_length - long.shape[1], 1)),
                ),
                axis=1,
            )
            speed_vector[k, :, :] = torch.cat(
                (
                    speed,
                    (speed[:, -1, :])
                    * torch.ones(size=(1, max_length - speed.shape[1], 1)),
                ),
                axis=1,
            )
            lateral[k, :, :] = torch.cat(
                (
                    lat,
                    (lat[:, -1, :]) * torch.ones(size=(1, max_length - lat.shape[1], 1)),
                ),
                axis=1,
            )
            relative_speed[k, :, :] = torch.cat(
                (
                    relative_s,
                    (relative_s[:, -1, :])
                    * torch.ones(size=(1, max_length - relative_s.shape[1], 1)),
                ),
                axis=1,
            )
    if experiment == "pedestrian_cphs":
        # print(acceleration.unsqueeze(-1).shape)
        return (
            (
                (distance.unsqueeze(-1)),
                ((acceleration.unsqueeze(-1), jerk.unsqueeze(-1))),
            ),
            (position.unsqueeze(-1), position.unsqueeze(-1)),
        )

    if experiment == "pedestrian":
        return (
            (distance.unsqueeze(-1)),
            (acceleration.unsqueeze(-1), jerk.unsqueeze(-1)),
        )
    elif experiment == "overtake":
        return (
            (distance.unsqueeze(-1), speed_vector.unsqueeze(-1)),
            (
                (lateral.unsqueeze(-1), longitudinal.unsqueeze(-1)),
                (relative_speed.unsqueeze(-1)),
            ),
        )


def get_final_w_set(formula, idx):
    """
    Returns the weight valuation specified by its index in sample set.

    Args:
        formula (WSTL_formula): Formula with associated weight valuations.
        idx (int): Index of the specified weight valuation.

    Returns:
        dict: Weight valuation at the specified index.
    """
    final_w = {}
    for key in formula.weights.keys():
        final_w[key] = formula.weights[key][:, idx].unsqueeze(-1)

    return final_w


def remove_true_sample(formula, w_idx):
    """
    Removes selected weight valuation sample from the set of samples.

    Args:
        formula (WSTL_formula): Formula containing weight valuations.
        w_idx (int): Index of the weight valuation sample to be removed.

    Returns:
        WSTL_formula: Updated formula after removing the specified sample.
    """

    for key in formula.weights.keys():
        formula.weights[key] = torch.cat(
            (formula.weights[key][:, :w_idx], formula.weights[key][:, w_idx + 1 :]),
            axis=1,
        )
    formula.update_weights()
    return formula


def get_signals_from_demonstrations(participant, demo_idx, ):
    """
    Prepares a tuple of signals suitable for WSTL formula robustness computation.

    Args:
        data (list): Raw signal data.
        experiment (str): Name of the scenario.
        N (int, optional): Number of signals to set.
        max_length (int, optional): Maximum length of signals.

    Returns:
        tuple: Tuple of signals formatted for WSTL formula computation.
    """
    ego_data = []
    ado_data = []

    N = len(demo_idx)
    for d in demo_idx:
        if d == 0 :
            data = pd.read_csv(f"./demonstrations/{participant}_trajectory-a.csv")
        else:
            data = pd.read_csv(f"./demonstrations/{participant}_trajectory-a_{d}.csv")
        ego_data.append(torch.tensor(np.array([data['x'][1::6], data['y'][1::6]]).T))
        ado_data.append(torch.tensor(np.array([data['x_ped'][1::6], data['y_ped'][1::6]]).T))

    filt = (
        1 / (20) * torch.ones((1, 1, 20))
    ) 
    max_length = max([len(ego_data[k]) for k in range(N)])

    acceleration = torch.ones(size=(N, max_length, 1))
    jerk = torch.ones(size=(N, max_length, 1))
    distance = torch.ones(size=(N, max_length, 1))
    position = torch.ones(size=(N, max_length, 1))

    for k in range(N):
        print(ego_data[k].shape)
        velocity = (ego_data[k][:-1, :] - ego_data[k][1:, :]) / 0.1
        print(velocity.shape)
        speed = torch.linalg.norm(velocity, axis=-1, keepdim=True)

        pos = ego_data[k][:, 0].unsqueeze(0).unsqueeze(-1)
        print(pos.shape)
        speed = speed.unsqueeze(0).unsqueeze(0).squeeze(-1)
        
        # speed_smooth = speed.unsqueeze(-1).squeeze(0)
        # print(speed_smooth.shape)
        speed_smooth = (
            torch.nn.functional.conv1d(speed, filt).squeeze(0).unsqueeze(-1)
        )
        accel = torch.linalg.norm(
            (speed_smooth[:, :-1, :] - speed_smooth[:, 1:, :]) / 0.1,
            axis=-1,
            keepdim=True,
        )
        jrk = torch.linalg.norm(
            (accel[:, :-1, :] - accel[:, 1:, :]) / 0.1, axis=-1, keepdim=True
        )

        dist = torch.linalg.norm(
            ego_data[k] - ado_data[k], axis=-1, keepdim=True
        ).unsqueeze(0)

        distance[k, :, :] = torch.cat(
            (
                dist,
                (dist[:, -1, :])
                * torch.ones(size=(1, max_length - dist.shape[1], 1)),
            ),
            axis=1,
        )
        acceleration[k, :, :] = torch.cat(
            (
                accel,
                (accel[:, -1, :])
                * torch.ones(size=(1, max_length - accel.shape[1], 1)),
            ),
            axis=1,
        )
        jerk[k, :, :] = torch.cat(
            (
                jrk,
                (jrk[:, -1, :]) * torch.ones(size=(1, max_length - jrk.shape[1], 1)),
            ),
            axis=1,
        )

        position[k, :, :] = torch.cat(
            (
                pos,
                (pos[:, -1, :]) * torch.ones(size=(1, max_length - pos.shape[1], 1)),
            ),
            axis=1,
        )
    
    return (
            (
                (distance.unsqueeze(-1)),
                ((acceleration.unsqueeze(-1), jerk.unsqueeze(-1))),
            ),
            (position.unsqueeze(-1), position.unsqueeze(-1)),
        )