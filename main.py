# Importing Libraries
import os
import copy
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Function

# Custom Libraries
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--lr',default= 1.2e-3, type=float, help='Learning rate')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--prune_freq', default=10, type=int)
parser.add_argument('--batch_size', default=60, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='mnist', type=str, help='mnist | cifar10 | fashionmnist | cifar100')
parser.add_argument('--arch_type', default='fc1', type=str, help='SmallVGG | binaryLeNet5 | binaryFc1 | fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121')
parser.add_argument('--prune_type', default='lt', type=str, help='lt | reinit')
parser.add_argument('--prune_percent', default=5, type=int, help='Pruning percent')
parser.add_argument('--binarize', action='store_true', help='binarize')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

writer = SummaryWriter()
sns.set_style('darkgrid')
best_accuracy = 0

# Function for Training
def train(model, train_loader, optimizer, criterion, mask, score):
    global best_accuracy
    train_ite = len(train_loader)
    compress = []
    bestacc = []
    loss = []
    acc = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    pbar = tqdm(train_loader)
    for i, (imgs, targets) in enumerate(pbar):
        imgs, targets = imgs.to(device), targets.to(device)
        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        comp1 = utils.print_nonzeros(model)
        
        if args.binarize:
            W = []
            cnt = 0
            for name, p in model.named_parameters():
                if 'weight' in name:
                    M = torch.from_numpy(mask[cnt]).to(device)
                    alpha = (torch.sum((p.data * M)**2) / torch.sum(M**2))**0.5
                    W.append(p.clone().detach())
                    p.data.sign_().mul_(M).mul_(alpha)
                    cnt += 1
            output = model(imgs)
            cnt = 0
            for name, p in model.named_parameters():
                if 'weight' in name:
                    p.data.zero_().add_(W[cnt])
                    cnt += 1
        else:
            output = model(imgs)
        
        train_loss = criterion(output, targets)
        train_loss.backward()
        cnt = 0
        for name, p in model.named_parameters():
            if 'weight' in name:
                #M = torch.from_numpy(mask[cnt]).to(device)
                #alpha = (torch.sum((p.data * M)**2) / torch.sum(M**2))**0.5
                
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < 1e-6, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
                #score[cnt] -= lr * p.grad.data
                cnt += 1
        
        optimizer.step()
        if i%args.prune_freq==args.prune_freq-1:
            cnt = 0
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    #tensor = score[cnt].cpu().numpy()
                    alive = tensor[np.nonzero(tensor)]
                    percentile_value = np.percentile(abs(alive), args.prune_percent)
                    
                    mask[cnt] = np.where(abs(tensor) < percentile_value, 0, mask[cnt])
                    p.data = torch.from_numpy(p.data.cpu().numpy() * mask[cnt]).to(p.device)
                    #p.data = torch.from_numpy(score[cnt].cpu().numpy() * mask[cnt]).to(p.device)
                    cnt += 1
            loss.append(train_loss.item())
            
            accuracy = test(model, mask, test_loader, criterion)
            acc.append(accuracy)
            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.checkdir(f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/')
                torch.save(model,f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{i}_model_{args.prune_type}.pth.tar')
            #print(f'Train Epoch: {i}/{train_ite} Loss: {train_loss.item():.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
            #pbar.set_description(f'Train Epoch: {i}/{train_ite} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       
            compress.append(comp1)
            bestacc.append(best_accuracy)
    writer.add_scalar('Accuracy/test', best_accuracy, comp1)
    # Dumping mask
    utils.checkdir(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/')
    with open(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl', 'wb') as fp:
        pickle.dump(mask, fp)
    return comp1, loss, acc, compress, bestacc
    

# Function for Testing
def test(model, mask, test_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if args.binarize:
                W = []
                cnt = 0
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        M = torch.from_numpy(mask[cnt]).to(device)
                        alpha = (torch.sum((p.data * M)**2) / torch.sum(M**2))**0.5
                        W.append(p.clone().detach())
                        p.data.sign_().mul_(M).mul_(alpha)
                        cnt += 1
                output = model(data)
                cnt = 0
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        p.data.zero_().add_(W[cnt])
                        cnt += 1
            else:
                output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def prune_by_percentile(model, mask, percent, reinit=False, **kwargs):
    cnt = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            percentile_value = np.percentile(abs(alive), percent)
            
            mask[cnt] = np.where(abs(tensor) < percentile_value, 0, mask[cnt])
            p.data = torch.from_numpy(tensor * mask[cnt]).to(p.device)
            cnt += 1
    return mask

def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if hasattr(m.weight, 'data'):
            init.normal_(m.weight.data, mean=1, std=0.02)
        if hasattr(m.bias, 'data'):
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reinit = True if args.prune_type=='reinit' else False

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == 'mnist':
        traindataset = datasets.MNIST('data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('data', train=False, transform=transform)
        from models.mnist import AlexNet, LeNet5, fc1, vgg, resnet, binaryFc1, binaryLeNet5
    elif args.dataset == 'cifar10':
        traindataset = datasets.CIFAR10('data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('data', train=False, transform=transform)
        from models.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, binaryFc1, SmallVGG
    elif args.dataset == 'fashionmnist':
        traindataset = datasets.FashionMNIST('data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('data', train=False, transform=transform)
        from models.mnist import AlexNet, LeNet5, fc1, vgg, resnet
    elif args.dataset == 'cifar100':
        traindataset = datasets.CIFAR100('data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('data', train=False, transform=transform)
        from models.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet
    else:
        raise Exception('\nWrong Dataset choice\n')

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)
    
    # Importing Network Architecture
    global model
    if args.arch_type == 'fc1':
       model = fc1.fc1().to(device)
    elif args.arch_type == 'lenet5':
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == 'alexnet':
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == 'vgg16':
        model = vgg.vgg16().to(device)
    elif args.arch_type == 'resnet18':
        model = resnet.resnet18().to(device)
    elif args.arch_type == 'densenet121':
        model = densenet.densenet121().to(device)
    elif args.arch_type == 'binaryFc1':
        model = binaryFc1.fc1().to(device)
    elif args.arch_type == 'binaryLeNet5':
        model = binaryLeNet5.LeNet5().to(device)
    elif args.arch_type == 'SmallVGG':
        model = SmallVGG.SmallVGG().to(device)
    else:
        raise Exception('\nWrong Model choice\n')

    # Weight Initialization
    model.apply(weight_init)
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/')
    torch.save(model, f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar')

    # Making Initial Mask
    mask = []
    score = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            mask.append(np.ones_like(tensor))
            #score.append(init.xavier_normal_(torch.ones_like(p.data)))
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    loss = []
    acc = []
    comp_ratio = []
    bestacc = []
    comp1 = 0
    for i in range(args.epochs):
        comp1, loss_, acc_, comp_ratio_, bestacc_ = train(model, train_loader, optimizer, criterion, mask, score)
        loss += loss_
        acc += acc_
        comp_ratio += comp_ratio_
        bestacc += bestacc_
    len_acc = len(acc)
    len_bestacc = len(bestacc)
    loss = np.array(loss)
    acc = np.array(acc)
    comp_ratio = np.array(comp_ratio)
    bestacc = np.array(bestacc)
    
    # Dumping Values for Plotting
    utils.checkdir(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/')
    loss.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat')
    acc.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat')
    comp_ratio.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat')
    bestacc.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat')

    # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
    plt.plot(np.arange(len_acc), 100*(loss - np.min(loss)).astype(float), c='blue', label='Loss')
    plt.plot(np.arange(len_acc), acc, c='red', label='Accuracy')
    plt.title(f'Loss and Accuracy ({args.dataset},{args.arch_type})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss and Accuracy')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color='gray')
    utils.checkdir(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/')
    plt.savefig(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png', dpi=1200)
    plt.close()
    
    plt.plot(comp_ratio, bestacc, c='blue', label='Winning tickets')
    plt.title(f'Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})')
    plt.xlabel('Unpruned Weights Percentage')
    plt.ylabel('test accuracy')
    plt.xlim(100, np.min(comp_ratio))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color='gray')
    utils.checkdir(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/')
    plt.savefig(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png', dpi=1200)
    plt.close()
    
