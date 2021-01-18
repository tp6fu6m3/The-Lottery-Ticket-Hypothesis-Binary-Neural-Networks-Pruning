# The-Lottery-Ticket-Hypothesis-Binary-Neural-Networks-Pruning 
[![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() 

This repository contains a Pytorch implementation of the paper [Multi-prize Lottery Ticket Hypothesis](https://openreview.net/pdf?id=U_mat0b9iv).

## Requirements
```
pip3 install -r requirements.txt
```
## Quick Start
### Using datasets/models included with this repository :
```
python3 main.py
```
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--epochs`	: Number of cycle of pruning that should be done. 
	- Default : `50`
- `--test_freq`	: Frequency for Validation 
	- Default : `50`
- `--batch_size`	: Batch size 
	- Default : `60`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `cifar10`
	- Default : `mnist`
- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5, `AlexNet` - AlexNet, `resnet18` - Resnet18, `vgg16` - VGG16 
	- Default : `fc1`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `5`
- `--mini_batch`	: Experiment on mini-batch
	- Default : `False`
- `--score`	: Using score matrix to determine the pruning mask
	- Default : `False`
- `--binarize`	: Model binarization
	- Default : `False`

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

## Acknowledgement 
Parts of code were borrowed from [Lottery Ticket Hypothesis in Pytorch](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch).
