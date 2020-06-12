import torch
from torchvision import transforms, datasets
from torch import nn
import time
import torchvision.models as models
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from load_data import data_loader
from collections import OrderedDict
from handle_command_line import parse_train
import sys
argv = sys.argv[1:]
data_dir, epoch = parse_train(argv)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_feat = model.fc.in_features
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_feat,len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.fc = classifier
model = model.to(device)
criterion = nn.NLLLoss()    
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.002, momentum=0.9)

def train_model(num_epochs=1):
    best_val_accu = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0.0
        since = time.time()
        trainloaders, train_size, _ = data_loader('train', data_dir)
        for inputs, labels in trainloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        time_elapsed = time.time() - since
        
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        validloaders, valid_size, _ = data_loader('valid', data_dir)
        for inputs, labels in validloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels.data)
        if val_correct.item()/valid_size > best_val_accu:
            best_val_accu = val_correct.item()/valid_size
            best_model_wts = copy.deepcopy(model.state_dict())
            
        print('training took:{}'.format(time_elapsed))
        print('train loss:{:.4f}'.format(running_loss/train_size))
        print('train accuracy:{:.4f}\n'.format((running_corrects.item()/train_size)))
        
        print('validation loss:{:.4f}'.format(val_loss/valid_size))
        print('validation accuracy:{:.4f}\n'.format(val_correct.item()/valid_size))
    model.load_state_dict(best_model_wts)
    print('best accuracy: {:.4f}'.format(best_val_accu))
    return model

model = train_model(epoch)

model.eval()
test_loss = 0
test_correct = 0
model = model.to(device)
testloaders, test_size, class_to_idx = data_loader('test',data_dir)
for inputs, labels in testloaders:          
    inputs = inputs.to(device)
    labels = labels.to(device)
    #print(inputs.shape)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)
    test_loss += loss.item() * inputs.size(0)
    test_correct += torch.sum(preds == labels.data)
        
print('test loss:{:.4f}'.format(test_loss/test_size))
print('test accuracy:{:.4f}\n'.format(test_correct.item()/test_size))

check_point={'class_to_idx':class_to_idx,
             'state_dict': model.state_dict(),
             'model': models.resnet18(pretrained=True),
             'classifier':classifier,
             }
torch.save(check_point,'check_point.pth')