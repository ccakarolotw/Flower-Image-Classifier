import torch
from PIL import Image
import numpy as np

def load_rebuild(file_name):
    cpt = torch.load(file_name)
    model = cpt['model']
    model.fc = cpt['classifier']
    model.load_state_dict(cpt['state_dict'])
    class_to_idx = cpt['class_to_idx']
    model.eval()
    return model, class_to_idx

def process_image(image):
    im = Image.open(image)
    
    width, height = im.size
    if width < height:
        im = im.resize((256, int(256*height/width)))
        width, height = 256, int(256*height/width)
    else:
        im = im.resize((int(256*width/height),256))
        width, height = int(256*width/height),256
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im = im.crop((left, top, right, bottom))
    
    np_im = np.array(im)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_im = (np_im - mean)/std
    np_im = np.transpose(np_im)
    return torch.tensor(np_im)