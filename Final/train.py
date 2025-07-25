import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import os

import notebookFunctions


ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store_true", default=False)
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', choices=['vgg16', 'densenet121', 'alexnet'], dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
gpu = pa.gpu
epochs = pa.epochs

trainloader, validloader, testloader = notebookFunctions.loadData(where)


model, criterion, optimizer = notebookFunctions.initializeModel(structure, hidden_layer1, dropout, lr)


notebookFunctions.trainModel(model, structure, trainloader, validloader, optimizer, criterion, epochs, gpu)

trainset,_,_ = notebookFunctions.loadDatasets(where)

notebookFunctions.save_checkpoint(model, structure, trainset, path, hidden_layer1, dropout, lr, epochs)

print("***CONSOLE***\n\n[*] The Model has been trained!")
