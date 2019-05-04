# Imports here

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import numpy as np
import json
import data_processing
import build_train_test
import argparse

parser1=argparse.ArgumentParser(description='Specify file path for data')
parser1.add_argument('path',type=str)
args1=parser1.parse_args()

parser2=argparse.ArgumentParser(description='Specify optional model training arguments')
parser2.add_argument('--save_dir',default='path',help='specify directory to save model')
parser2.add_argument('--arch',default='models.resnet50',help='specify pretrained models from pytorch')
parser2.add_argument('--learning_rate',default=0.01)
parser2.add_argument('--epochs',default=10)
parser2.add_argument('--hidden_units',default=1024)
parser2.add_argument('--device',default='cpu')
args2=parser2.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

train_loader,valid_loader,test_loader=data_processing.data_trans(args1.path)
model=build_train_test.building_and_training(args2.arch,args2.hidden_units,args2.epochs,
                            args2.learning_rate,train_loader,valid_loader,args2.device)
build_train_test.testing_model(model,test_loader)
build_train_test.save_checkpoint(model,train_dir,args2.save_dir)
build_train_test.load_checkpoint(args2.save_dir)
