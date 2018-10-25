from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from time import gmtime, strftime
import os
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
# model = Net( growthRate=16, depth=5, reduction=0.5, bottleneck=True)
# model = Net()

model = Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (torch.cuda.is_available())
model = model.to(device)
print ('Model Loaded')
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

model.apply(weight_init)
print ('Weight Changes')
optimizer  = optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
print ('Optimizer Done')
def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\n Training Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():    
            data, target = Variable(data), Variable(target)
            data = data.to(device)
            target = target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

millis = int(round(time.time() * 1000))
model_directory = 'Experiment_Trained'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

current_model_directory = 'October_20_' + str(millis)
if not os.path.exists(model_directory+"/"+current_model_directory):
        os.makedirs(model_directory+"/"+current_model_directory)

def save_checkpoint(state, filename):
    torch.save(state, filename)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = model_directory + "/" + current_model_directory + "/"  + 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
    lr  = (0.95**epoch)*args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
