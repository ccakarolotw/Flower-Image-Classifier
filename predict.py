from predict_utility import load_rebuild, process_image
import torch
from PIL import Image
import numpy as np
import os
import json
from handle_command_line import parse_predict
import matplotlib.pyplot as plt
import sys
argv = sys.argv[1:]
checkpoint, image_path = parse_predict(argv)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
model, class_to_idx = load_rebuild(checkpoint)
idx_to_class = {v: k for k, v in class_to_idx.items()} 

def predict(image_path, model, topk=5):
    device = torch.device("cpu")
    model = model.to(device)

    img = process_image(image_path)
    img = torch.reshape(img,(1,3,224,224))
    img = img.float()
    out = model(img)
    probs, labels = torch.exp(out).topk(topk)
    classes = [idx_to_class[label] for label in labels.numpy()[0]]
    name = [cat_to_name[clas] for clas in classes]
    print('Top 5 predictions and probabilities:')
    for i in range(5):
        print('{}: {}'.format(name[i],probs.detach().numpy()[0][i]))
    return probs, name
    
predict(image_path,model)
    