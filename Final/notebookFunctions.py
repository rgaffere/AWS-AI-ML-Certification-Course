import torch
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def loadData(data_dir='flowers'):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225]),])

    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    testset = datasets.ImageFolder(test_dir, transform=test_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False)
    
    return trainloader, testloader, validloader

def loadDatasets(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225]),])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    return train_dataset, test_dataset, validation_dataset

def initializeModel(structure, hiddenLayer, dropout=0.5, lr=.003):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)       
        inputLayer = 25088
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputLayer = 1024
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
        inputLayer = 9216
    else:
        print("{} is not a valid model.\nOptions: vgg16,densenet121,or alexnet?".format(structure))

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(inputLayer, hiddenLayer),
                                    nn.ReLU(),
                                    nn.Dropout(.5),
                                    nn.Linear(hiddenLayer, 120),
                                    nn.ReLU(),
                                    nn.Dropout(.5),
                                    nn.Linear(120, 80),
                                    nn.ReLU(),
                                    nn.Dropout(.5),
                                    nn.Linear(80, 102),
                                    nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

def trainModel(model, structure, trainloader, validloader, optimizer, criterion, epochs=7, gpu = False):
    inputs, labels = next(iter(trainloader))
    inputs2, labels2 = next(iter(validloader))

    epochs = epochs
    print_every = 5
    steps = 0
    loss_show=[]

    if torch.cuda.is_available() and gpu:
            model.cuda()
            
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            if torch.cuda.is_available() and gpu:
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
            
            
                for inputs2, labels2 in validloader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available() and gpu:
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                vlost = vlost / len(validloader)
                accuracy = accuracy /len(validloader)
            
                    
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}".format(accuracy))
            
                running_loss = 0

def save_checkpoint(model, structure, trainset, path, hiddenLayer, dropout=0.5, lr=0.001, epochs=7, gpu=False):
    if structure == 'vgg16':
        inputLayer = 25088
    elif structure == 'densenet121':
        inputLayer = 1024
    elif structure == 'alexnet':
        inputLayer = 9216
        
    model.class_to_idx = trainset.class_to_idx
    model.cpu()
    checkpoint = {
                  'input_size': hiddenLayer,
                  'structure_out': inputLayer,
                  'output_size': 102,
                  'state_dict': model.classifier.state_dict(),
                  'dropout': dropout,
                  'lr': lr,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'structure': structure}
    torch.save(checkpoint, path)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(src, dropout=.5, gpu = False):
    if not gpu:
        checkpoint = torch.load(src, map_location='cpu')
    else:
        checkpoint = torch.load(src)
    fc1 = checkpoint['input_size']
    
    if checkpoint['structure'] == 'vgg16':
        model = models.vgg16(pretrained=True)       
    elif checkpoint['structure'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['structure'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    model.classifier = nn.Sequential(nn.Linear(checkpoint['structure_out'], fc1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fc1, 120),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(120, 80),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(80, 102),
                                nn.LogSoftmax(dim=1))
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier.load_state_dict(checkpoint['state_dict'])

    return model
       
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(image)
    
    img_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    
    img_out = img_transform(pil_img)
    return img_out

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, gpu=False, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    if torch.cuda.is_available() and gpu:
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
