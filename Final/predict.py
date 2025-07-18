import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import os

import notebookFunctions

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
# GPU is not enabled automatically so you have to specify if you want to use it
ap.add_argument('--gpu', default=False, action="store_true", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
gpu = pa.gpu
input_img = pa.input_img
path = pa.checkpoint
cat = pa.category_names

with open(os.path.join('.', cat), 'r') as json_file:
    cat_to_name = json.load(json_file)
    
trainloader, validloader, testloader = notebookFunctions.loadData()
model = notebookFunctions.load_model(path)

probabilities = notebookFunctions.predict(path_image, model, gpu)

probabilities[1][0].cpu()
probabilities[0][0].cpu()
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

print("\n\n***OUTPUTS***")
i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1