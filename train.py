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
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

train_loader,valid_loader,test_loader=data_processing.data_trans(args1.path)
