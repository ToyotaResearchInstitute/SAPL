# SAPL: Safe Active Preference Learning

This repository contains the source code and experiments for the Safe Active Preference Learning (SAPL) framework with Signal Temporal Logic.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Human Subject Study](#human-subject-study)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The SAPL framework is designed to perform active preference learning within safe behaviors. It leverages Signal Temporal Logic to ensure safety during the learning process. This repository includes the source code, empirical experiments, and tools for conducting human subject studies.

## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```
**Remark:** It requires Python3.10 or above.

## Usage
### Tutorial
To get started with the SAPL framework, you can use the ```tutorial.ipynb``` script. This script provides a step-by-step guide to using the SAPL framework.

### Running Empirical Experiments
You can run various empirical experiments provided in the ```experiments``` directory. For example, to run the convergence analysis when the true weight valuation is out of distribution, use:
```sh
python experiments/convergence_analysis_ood.py
```

### Human Subject Study
To start a human subject study, you can use the ```SAPL_human_subject_study.py``` script. This script allows you to show trajectories to participants and ask validation questions at the end of the experiment.

## Experiments
The ```experiments``` directory contains several scripts for different types of experiments:

- ```convergence_analysis_ood.py```: Convergence analysis with out-of-distribution true weight valuation.
- ```convergence_analysis_unlimited_q.py```: Convergence analysis with unlimited questions.
- ```convergence_w_different_u.py```: Convergence with different utility functions.
- ```noisy_convergence.py```: Convergence analysis with noisy data.
- ```query_selection_vs_random.py```: Comparison of query selection strategies versus random selection.
- ```safety_guarantees.py```: Analysis of safety guarantees.

## Human Subject Study
In human subject studies, you need a means to show trajectories to participants. This can be done using a simulator or by showing videos. The simulator variable in the script can be set to choose the appropriate option. For simplicity, the provided script uses the no simulator option.

## Contributing
We welcome contributions to the SAPL framework. Please follow the guidelines in the ```.pre-commit-config.yaml``` file to ensure code quality. To set up pre-commit hooks, run:
```sh
pre-commit install
```

## License
This project is licensed under the Creative Commons Attribution-NonCommercial license. See the LICENSE file for details.

## Citation
If you use the SAPL framework in your research, please cite the following paper:
```
@inproceedings{Karagulle2024SAPL,
author = {Karagulle, Ruya and Ozay, Necmiye and Arechiga, Nikos and Decastro, Jonathan and Best, Andrew},
title = {Incorporating Logic in Online Preference Learning for Safe Personalization of Autonomous Vehicles},
year = {2024},
doi = {10.1145/3641513.3650129},
booktitle = {Proceedings of the 27th ACM International Conference on Hybrid Systems: Computation and Control},
articleno = {5},
numpages = {11},
location = {Hong Kong SAR, China},
series = {HSCC '24}
}
```


