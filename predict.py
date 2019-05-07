import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import json
import argparse
import functions_for_predict

parser=argparse.ArgumentParser(description='Specify file path for model')
parser.add_argument('path',default=os.getcwd())
parser.add_argument('path_to_image')
parser.add_argument('--topk',default=5)
parser.add_argument('--device',default='gpu')
args=parser.parse_args()


model=functions_for_predict.load_checkpoint(args.path).to(args.device)
np_image=functions_for_predict.process_image(args.path_to_image)
probs,classes=functions_for_predict.predict(np_image, model, args.topk,args.device)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

probs,classes =probs.detach().numpy(),classes.detach().numpy()
classes=classes.astype('str')
names=[]
for i in range(5):
    name=cat_to_name.get(classes[0,i])
    names.append(name)
for name,prob in zip(names,probs):
    print({name:prob})
