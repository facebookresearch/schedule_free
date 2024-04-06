
import os
import numpy as np
import random
import itertools
import argparse
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10

import schedulefree

def set_seed(seed: int = 42, is_deterministic=False) -> None:

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if is_deterministic:
        print("This ran")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class CIFAR10Harness:
    def __init__(self, args):
        self.args = args
        if args.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu"))
        else:
            self.device = torch.device(self.args.device)

        self.model = self.create_model()
        self.create_optimizer()
        self.train_loader, self.test_loader = self.create_data_loaders()
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self):
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(self.device)
        return model

    def create_optimizer(self):
        if self.args.optimizer_variant == 'sgd':
            print(f'opitmizer params: {args.warmup_steps}, {args.lr}, {args.momentum}, {args.wd}')
            self.optimizer = schedulefree.SGDScheduleFree(
                self.model.parameters(),
                warmup_steps=self.args.warmup_steps,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.wd,
            )

        elif self.args.optimizer_variant == 'adam':
            self.optimizer = schedulefree.AdamWScheduleFree(
                self.model.parameters(),
                warmup_steps=self.args.warmup_steps,
                lr=self.args.lr,
                weight_decay=self.args.wd,
            ) 

    def create_data_loaders(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        testset = CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, test_loader


    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.train()
        train_loss = 0
        correct = 0
        total = 0
        with tqdm.tqdm(self.train_loader, unit='batch') as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f'Epoch: {epoch}')
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if args.dry_run:
                    break
        train_loss /= len(self.train_loader)
        accuracy = 100. * (correct / total)

        return train_loss, accuracy

    def test(self):
        self.model.train()
        self.optimizer.eval()
        with torch.no_grad():
            for batch in itertools.islice(self.train_loader, 50):
                _ = self.model(batch[0].to(self.device))
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(self.test_loader, leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        test_loss /= len(self.test_loader)
        accuracy = 100. * (correct / total)
        
        return test_loss, accuracy

def train_experiment(args):
    set_seed(args.seed)
    if args.save_to_dataframe:
        data_df = {}
        data_df['epoch'] = []
        data_df['train_acc'] = []
        data_df['train_loss'] = []
        data_df['test_acc'] = []
        data_df['test_loss'] = []

    harness = CIFAR10Harness(args)

    for epoch in range(args.epochs):
        train_loss, train_acc = harness.train_one_epoch(epoch)
        test_loss, test_acc = harness.test()
        if args.save_to_dataframe:
            data_df['epoch'].append(epoch)
            data_df['train_acc'].append(train_acc)
            data_df['train_loss'].append(train_loss)
            data_df['test_acc'].append(test_acc)
            data_df['test_loss'].append(test_loss)
            pd.DataFrame(data_df).to_csv('metrics.csv', index=False)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

    return harness.model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Schedule-Free PyTorch experiments with CIFAR10')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--optimizer_variant', type=str, default='sgd', help='Optimizer variant to use')
    parser.add_argument('--dry_run', action='store_true', help='Dry run')
    parser.add_argument('--save-model', action='store_true', help='Save model')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--warmup_steps', type=int, default=30, help='Warmup steps')
    parser.add_argument('--lr', type=float, default=10, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum (for SGD)')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save_to_dataframe', action='store_true', help='Save data to dataframe')

    args = parser.parse_args()

    final_model = train_experiment(args)

    if args.save_model:
        torch.save(final_model.state_dict(), 'cifar10_model.pth')

