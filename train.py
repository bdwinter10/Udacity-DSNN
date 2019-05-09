# Imports here
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import numpy as np
import data_processing
import build_train_test
import argparse

parser1=argparse.ArgumentParser(description='Specify file path for data')
parser1.add_argument('path',type=str)
parser1.add_argument('--save_dir',default=os.getcwd(),help='specify directory to save model')
parser1.add_argument('--arch',default="resnet50",help='specify pretrained models from pytorch')
parser1.add_argument('--learning_rate',default=0.001)
parser1.add_argument('--epochs',default=1)
parser1.add_argument('--hidden_units',default=1024)
parser1.add_argument('--device',default='gpu')
args1=parser1.parse_args()

train_dir,valid_dir,test_dir,train_loader,valid_loader,test_loader=data_processing.data_trans(args1.path)
model,classifier,criterion,optimizer=build_train_test.building(args1.arch,args1.hidden_units,
                                                               args1.learning_rate,args1.device)
model=build_train_test.training(model,args1.epochs,criterion,optimizer,train_loader,valid_loader,args1.device)
build_train_test.testing_model(model,test_loader,criterion,args1.device)
build_train_test.save_checkpoint(model,criterion,optimizer,train_dir,args1.save_dir)
