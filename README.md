# The Lottery Ticket Hypothesis:BNN Pruning 
[![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() 

This repository contains a Pytorch implementation of the paper [Multi-prize Lottery Ticket Hypothesis](https://openreview.net/pdf?id=U_mat0b9iv).

## Quick Start
### Using datasets/models included with this repository :
```
python3 main.py
```
- `--lr`		: Learning rate (Default : `1.2e-3`)
- `--prune_ite`		: Number of cycle of pruning that should be done (Default : `30`)
- `--train_ite`		: Number of cycle of training per pruning (Default : `100`)
- `--test_freq`		: Frequency for Validation (Default : `10`)
- `--batch_size`	: Batch size (Default : `60`)
- `--dataset`		: Choice of dataset (Options : `mnist`, `cifar10`, Default : `mnist`)
- `--arch_type`		: Type of architecture (Options : `fc1`, `lenet5`, `AlexNet`, `resnet18`, `vgg16`, Default : `fc1`)
- `--prune_percent`	: Percentage of weight to be pruned after each cycle (Default : `5`)
- `--mini_batch`	: Experiment on mini-batch (Default : `False`)
- `--score`		: Using score matrix to determine the pruning mask (Default : `False`)
- `--binarize`		: Model binarization (Default : `False`)
- `--reinit`		: Randomly re-initializing weight (Default : `False`)

## Repository Structure
```
The-Lottery-Ticket-Hypothesis-Binary-Neural-Networks-Pruning
├── models
│   ├── cifar10
│   │   ├── AlexNet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   ├── SmallVGG.py
│   │   └── vgg.py
│   └── mnist
│       ├── AlexNet.py
│       ├── fc1.py
│       ├── LeNet5.py
│       ├── resnet.py
│       ├── SmallVGG.py
│       └── vgg.py
├── dumps
├── main.py
├── plots
├── README.md
├── requirements.txt
├── saves
└── utils.py

```
