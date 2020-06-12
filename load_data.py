import torch
from torchvision import transforms, datasets
import numpy as np

def data_loader(state, data_dir = 'flowers'):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomResizedCrop((224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]),
                      'val_test' : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop((224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])}
    
    if state=='train':
        dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        size = len(dataset)
        class_to_idx = dataset.class_to_idx
    elif state=='valid':
        dataset = datasets.ImageFolder(root=valid_dir, transform=data_transforms['val_test'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        size = len(dataset)
        class_to_idx = dataset.class_to_idx
    elif state=='test':
        dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms['val_test'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        size = len(dataset)
        class_to_idx = dataset.class_to_idx
        
    return dataloader, size, class_to_idx
