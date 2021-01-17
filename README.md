# The-Lottery-Ticket-Hypothesis-Binary-Neural-Networks-Pruning 
[![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() 

This repository contains a Pytorch implementation of the paper [Multi-prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning a Randomly Weighted Network](https://openreview.net/pdf?id=U_mat0b9iv).

## Requirements
```
pip3 install -r requirements.txt
```
## Quick Start
### Using datasets/models included with this repository :
```
python3 main.py
```
- `--prune_type` : Type of pruning  
	- Options : `lt` - Lottery Ticket Hypothesis, `reinit` - Random reinitialization
	- Default : `lt`
- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5, `AlexNet` - AlexNet, `resnet18` - Resnet18, `vgg16` - VGG16 
	- Default : `fc1`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `fashionmnist`, `cifar10`, `cifar100` 
	- Default : `mnist`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `10`
- `--prune_iterations`	: Number of cycle of pruning that should be done. 
	- Default : `35`
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--batch_size`	: Batch size 
	- Default : `60`
- `--end_iter`	: Number of Epochs 
	- Default : `100`
- `--print_freq`	: Frequency for printing accuracy and loss 
	- Default : `1`
- `--valid_freq`	: Frequency for Validation 
	- Default : `1`
- `--gpu`	: Decide Which GPU the program should use 
	- Default : `0`


## Repository Structure
```
Lottery-Ticket-Hypothesis-in-Pytorch
├── archs
│   ├── cifar10
│   │   ├── AlexNet.py
│   │   ├── densenet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── cifar100
│   │   ├── AlexNet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   └── mnist
│       ├── AlexNet.py
│       ├── fc1.py
│       ├── LeNet5.py
│       ├── resnet.py
│       └── vgg.py
├── combine_plots.py
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
