"""
This module contains tools to ask a given question on the motion base simulator.
Motion base simulator uses CARLA and ROS2.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import csv


class ReplayerRunData:
    def __init__(self):
        self.left_ego_command = None
        self.left_ado_command = None
        self.right_ego_command = None
        self.right_ado_command = None
        self.left_start_ego_command = None
        self.left_start_ado_command = None
        self.right_start_ego_command = None
        self.right_start_ado_command = None


def load_commands(question, host):
    """Returns commands to eplayer data."""
    base_command = [
        "ros2",
        "run",
        "vehicle_replayer",
        "vehicle_replayer",
        "--ros-args",
        "-p",
        "trigger_type:=none",
        "-p",
        f"host:={host}",
        "-p",
        "settling_height:=0.15",
    ]

    replayer_data = ReplayerRunData()
    for side, key in [
        ["left", "bag1"],
        ["right", "bag2"],
    ]:
        ego_node = f"{side}_ego_replayer"
        ado_node = f"{side}_ado_replayer"
        # bag = question[key]
        start_time = question[f"{key}_start_sim"]
        end_time = question[f"{key}_end_sim"]
        ado_id = question[f"{key}_ado_id"]
        command = deepcopy(base_command) + [
            "-p",
            f"bag:={question[key]}",
            "-p",
            f"start_time:={start_time}",
            "-p",
            f"end_time:={end_time}",
        ]
        ego_command = deepcopy(command) + [
            "-p",
            "mode:=ego",
            "-r",
            f"__node:={ego_node}",
        ]
        ado_command = deepcopy(command) + [
            "-p",
            "mode:=spawn",
            "-p",
            "spawn_vehicle_type:=walker.pedestrian.0001",
            "-p",
            f"vehicle_id:={ado_id}",
            "-r",
            f"__node:={ado_node}",
        ]
        start_ego_command = [
            "ros2",
            "service",
            "call",
            f"/{ego_node}/resume",
            "motion_sim_interface/srv/Empty",
        ]
        start_ado_command = [
            "ros2",
            "service",
            "call",
            f"/{ado_node}/resume",
            "motion_sim_interface/srv/Empty",
        ]
        setattr(replayer_data, f"{side}_ego_command", ego_command)
        setattr(replayer_data, f"{side}_ado_command", ado_command)
        setattr(replayer_data, f"{side}_start_ego_command", start_ego_command)
        setattr(replayer_data, f"{side}_start_ado_command", start_ado_command)
    return replayer_data


def play_trajectory(replayer_data, left=True):
    # Start subprocesses for each of the replayers
    ego_command = (
        replayer_data.left_ego_command if left else replayer_data.right_ego_command
    )
    ado_command = (
        replayer_data.left_ado_command if left else replayer_data.right_ado_command
    )
    start_ego_command = (
        replayer_data.left_start_ego_command
        if left
        else replayer_data.right_start_ego_command
    )
    start_ado_command = (
        replayer_data.left_start_ado_command
        if left
        else replayer_data.right_start_ado_command
    )

    processes = []
    print(ego_command)
    processes.append(subprocess.Popen(ego_command, shell=False))
    print(ado_command)
    processes.append(subprocess.Popen(ado_command, shell=False))

    input("Press start when trajectories have loaded")
    print(start_ego_command)
    processes.append(subprocess.Popen(start_ego_command, shell=False))
    print(start_ado_command)
    processes.append(subprocess.Popen(start_ado_command, shell=False))
    for process in processes[::-1]:
        process.wait()
    print("Trajectories complete.")


def ask_a_question_interactive(question_csv, question_id, host):
    assert question_csv.exists(), f"invalid csv path {question_csv}"
    questions = {}
    with open(question_csv, "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            questions[int(row["Question_no"])] = row

    assert (
        question_id in questions
    ), f"Question {question_id} not in {list(questions.keys())}"

    question = questions[question_id]
    replayer_data = load_commands(question, host)

    response = None
    while response != -1 and response != "-1":
        print("Action:")
        print("- 1: Play left trajectory")
        print("- 2: Play right trajectory")
        print("- -1: quit")
        response = input("Choose: ")
        if response == 1 or response == "1":
            play_trajectory(replayer_data, True)
            pass
        elif response == 2 or response == "2":
            play_trajectory(replayer_data, False)
            pass


def main():
    parser = argparse.ArgumentParser("Repeat a replay question")
    parser.add_argument("question_csv")
    parser.add_argument("question_id", type=int)
    parser.add_argument("--host", default="10.110.21.114", required=False)
    args = parser.parse_args()
    ask_a_question_interactive(Path(args.question_csv), args.question_id, args.host)


if __name__ == "__main__":
    main()
