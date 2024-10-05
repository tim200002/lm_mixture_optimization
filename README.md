# Mixture Optimization for LLM Pre-Training Utilizing Bayesian Optimization

This repository contains the code from my research internship, which focuses on optimizing language mixtures during pre-training of a Language Model (LLM) using Bayesian Optimization. Conducted at the Chair of Machine Learning at the Technical University of Munich under the guidance of Prof. Heckel, this work is part of my Master's program in Electrical Engineering. The internship is a 12-credit module designed to prepare students for their Master’s thesis by completing a research project.

In this project, I explored a novel approach to optimize the data mixtures used in LLM pre-training by applying Bayesian Optimization on down-scaled proxy models. This method provided a computationally efficient way to identify promising mixtures, outperforming other techniques when normalized for training costs. However, scaling these mixtures from proxy models to production-scale models proved challenging, highlighting the need for further research in this area.

The final report is available [here](assets/report.pdf). The final presentation is available [here](assets/presentation.pptx).

## Abstract
To achieve broad knowledge across many tasks, language models (LM) are pre-trained with text from a wide range of different datasets/domains. The proportions of these datasets play a crucial role in determining LM performance. In this work, we introduce a novel approach to data-mixture optimization, formulating it as a sequential black-box optimization problem, solved using Bayesian Optimization (BO). BO is employed to leverage its sample efficiency, enabling effective exploration of the data mixture space while minimizing the number of training runs required. Instead of working on a full production scale, our method identifies optimal data mixtures on a down-scaled proxy model, which is more effective to train than the full-scale model, assuming that optimal mixture distributions remain consistent across model scales.  

Through extensive experiments, we demonstrate the effectiveness of our approach in identifying promising proxy-model mixtures, outperforming random search, several popular mixtures from the literature, and state-of-the-art (SOTA) mixture optimization methods. However, we also identify limitations in our approach, the biggest lying in the assumption of scale invariance, showing that transferring optimal mixtures from proxy models to larger models poses challenges, as some mixtures that perform poorly at small scales outperform our proxy mixtures when scaled up.  This highlights the need for further research into the scaling behavior of data mixtures to better harness the insights gained from small proxy models.

## Installation

### Environment Setup
A dockerfile to setup a proper environment is provided in the repository. Please build it or create a similar environment following the file.

### Open-LM
Our code for training is based on the [open-lm](https://github.com/mlfoundations/open_lm) training framework. Please install the open-lm package by following the instructions in the repository.

## Usage
In short this repo provides code to make conducting sequential black-box optimizzation of LLMs easy. You can think of it as my own implementaiton of the well-known Axios framework. In the `config` file you can setup the specific experiment setup (e.g., number of experiments, optimization procedure (e.g., bayesian or TuRbO), LLM training config, ...). The code is then executed by running the `main.py` or `run.sh` script.

At each trial, the optimizier will first propose a new data mxiture. Based on this mixture, we create a new dataset by aporporiatley sampling the original dataset. This temporary dataset is stored in a data location according to the config file. The LLM is then trained on this dataset and the performance is evaluated. The temporary dataset and other temporary checkpoints are deleted after each trial to save memory (you can overide this behaviour in the config file).

During trial the execution history, training checpoints, training logs, .. are stored in the `logs` directory. The stated stored in this directory allows for easy resuming of a paused experiment. Training logs are uploaded to wandb as well (please make sure to setup your local wandb appropriately).