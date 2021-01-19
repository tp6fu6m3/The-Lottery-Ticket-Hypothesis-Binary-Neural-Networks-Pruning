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
from torch.autograd import Function

# Custom Libraries
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default= 1.2e-3, type=float)
parser.add_argument('--prune_ite', default=10, type=int)
parser.add_argument('--train_ite', default=2, type=int)
parser.add_argument('--test_freq', default=2, type=int)
parser.add_argument('--batch_size', default=60, type=int)
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'], type=str)
parser.add_argument('--arch_type', default='Conv2', choices=['fc1', 'Conv2', 'Conv4', 'Conv6', 'Conv8', 'lenet5', 'alexnet', 'vgg16', 'resnet18', 'densenet121'], type=str)
parser.add_argument('--prune_percent', default=10, type=int, help='Pruning percent')
parser.add_argument('--mini_batch', action='store_true')
parser.add_argument('--score', action='store_true')
parser.add_argument('--binarize', action='store_true')
parser.add_argument('--reinit', action='store_true')
args = parser.parse_args()

sns.set_style('whitegrid')

# Function for Training
def train(model, train_loader, optimizer, criterion, mask, score):
    best_accuracy = 0
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
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < 1e-6, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
                score[cnt] -= lr * p.grad.data
                cnt += 1
        
        optimizer.step()
        if args.mini_batch or i%(len(train_loader)//args.test_freq)==0:
            if args.mini_batch:
                cnt = 0
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        if args.score:
                            if score[cnt].dim()>3:
                                sorted, indices = torch.sort(torch.abs(score[cnt]), dim=0)
                                tensor = indices.cpu().numpy()
                                percentile_value = np.percentile(tensor, args.prune_percent)
                                mask[cnt] = np.where(tensor < percentile_value, 0, mask[cnt])
                        else:
                            tensor = p.data.cpu().numpy()
                            alive = tensor[np.nonzero(tensor)]
                            percentile_value = np.percentile(abs(alive), args.prune_percent)
                            mask[cnt] = np.where(abs(tensor) < percentile_value, 0, mask[cnt])
                        cnt += 1
            loss.append(train_loss.item())
            
            accuracy = test(model, mask, test_loader, criterion)
            acc.append(accuracy)
            # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.checkdir(f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/')
                torch.save(model,f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{i}_model.pth.tar')
            #print(f'Train Epoch: {i}/{train_ite} Loss: {train_loss.item():.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
            #pbar.set_description(f'Train Epoch: {i}/{train_ite} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       
            if args.mini_batch:
                comp1 = utils.print_nonzeros(model)
                compress.append(comp1)
                bestacc.append(best_accuracy)
    utils.checkdir(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/')
    with open(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/mask_{comp1}.pkl', 'wb') as fp:
        pickle.dump(mask, fp)
    if args.mini_batch:
        return loss, acc, compress, bestacc
    else:
        return loss, acc, best_accuracy
    

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

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == 'mnist':
        traindataset = datasets.MNIST('data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('data', train=False, transform=transform)
        from models.mnist import AlexNet, LeNet5, fc1, vgg, resnet, SmallVGG
    elif args.dataset == 'cifar10':
        traindataset = datasets.CIFAR10('data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('data', train=False, transform=transform)
        from models.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, SmallVGG
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
    elif args.arch_type == 'Conv2':
        model = SmallVGG.Conv2().to(device)
    elif args.arch_type == 'Conv4':
        model = SmallVGG.Conv4().to(device)
    elif args.arch_type == 'Conv6':
        model = SmallVGG.Conv6().to(device)
    elif args.arch_type == 'Conv8':
        model = SmallVGG.Conv8().to(device)
    else:
        raise Exception('\nWrong Model choice\n')

    # Weight Initialization
    model.apply(weight_init)
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/')
    torch.save(model, f'{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict.pth.tar')


    # Making Initial Mask
    mask = []
    score = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            mask.append(np.ones_like(tensor))
            if p.data.dim()>1:
                score.append(init.xavier_normal_(torch.ones_like(p.data)))
            else:
                score.append(init.normal_(torch.ones_like(p.data), mean=1, std=0.02))
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    comp_ratio = []
    bestacc = []
    for i in range(args.prune_ite):
        if args.mini_batch:
            loss, acc, comp_ratio, bestacc = train(model, train_loader, optimizer, criterion, mask, score)
            comp1 = utils.print_nonzeros(model)
        else:
            loss = []
            acc = []
            for j in range(args.train_ite):
                loss_, acc_, best_accuracy_ = train(model, train_loader, optimizer, criterion, mask, score)
                loss += loss_
                acc += acc_
            cnt = 0
            for name, p in model.named_parameters():
                if 'weight' in name:
                    if args.score:
                        if score[cnt].dim()>3:
                            sorted, indices = torch.sort(torch.abs(score[cnt]), dim=0)
                            tensor = indices.cpu().numpy()
                            percentile_value = np.percentile(tensor, args.prune_percent)
                            mask[cnt] = np.where(tensor < percentile_value, 0, mask[cnt])
                    else:
                        tensor = p.data.cpu().numpy()
                        alive = tensor[np.nonzero(tensor)]
                        percentile_value = np.percentile(abs(alive), args.prune_percent)
                        mask[cnt] = np.where(abs(tensor) < percentile_value, 0, mask[cnt])
                    cnt += 1
            if args.reinit:
                model.apply(weight_init)
                cnt = 0
                for name, p in model.named_parameters(): 
                    if 'weight' in name: 
                        p.data = torch.from_numpy(mask[cnt] * p.data.cpu().numpy()).to(p.device)
                        cnt += 1
            else:
                cnt = 0
                for name, p in model.named_parameters(): 
                    if 'weight' in name: 
                        p.data = torch.from_numpy(mask[cnt] * initial_state_dict[name].cpu().numpy()).to(p.device)
                        cnt += 1
                    if 'bias' in name:
                        p.data = initial_state_dict[name]
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            comp1 = utils.print_nonzeros(model)
            comp_ratio.append(comp1)
            bestacc.append(best_accuracy_)
        
        len_acc = len(acc)
        loss = np.array(loss)
        acc = np.array(acc)
        
        # Dumping Values for Plotting
        utils.checkdir(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/')
        loss.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/all_loss_{comp1}.dat')
        acc.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/all_accuracy_{comp1}.dat')
        
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
        plt.savefig(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/LossVsAccuracy_{comp1}.png', dpi=1200)
        plt.close()
        
        if args.mini_batch:
            break
    
    comp_ratio = np.array(comp_ratio)
    bestacc = np.array(bestacc)
    print(bestacc)
    
    utils.checkdir(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/')
    comp_ratio.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/compression.dat')
    bestacc.dump(f'{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/bestaccuracy.dat')

    plt.plot(comp_ratio, bestacc, c='blue', label='Winning tickets')
    plt.title(f'Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})')
    plt.xlabel('Unpruned Weights Percentage')
    plt.ylabel('test accuracy')
    plt.xlim(np.max(comp_ratio), np.min(comp_ratio))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color='gray')
    utils.checkdir(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/')
    plt.savefig(f'{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/AccuracyVsWeights.png', dpi=1200)
    plt.close()
    
