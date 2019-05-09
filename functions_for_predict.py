import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import json
import fc_model

def load_checkpoint(path,map_location,hidden_units,learning_rate):
    model=models.resnet50()
    for para in model.parameters():
        para.requires_grad=False
    classifier=nn.Sequential(nn.Linear(2048, hidden_units,learning_rate),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256,102),
                             nn.LogSoftmax(dim=1))
    model.fc=classifier
    optimizer=optim.Adam(model.fc.parameters(), lr=learning_rate)
    checkpoint=torch.load(path+'/checkpoint.pth',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model

def process_image(path_to_image):
    im=PIL.Image.open(path_to_image)
    im.thumbnail((256,256))
    np_image=np.array(im)

    np_image=np_image/255.0
    np_image=np_image-[0.485,0.456,0.406]
    np_image=np_image/[0.229,0.224,0.225]
    np_image=np_image.transpose((2,0,1))
    np_image=torch.tensor(np_image)
#     print(np_image.shape,np_image.unsqueeze(0).shape)
    return np_image

def predict(np_image, model, topk,device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device=='cuda' else "cpu")
    inputs=np_image.unsqueeze(0)
    inputs=inputs.type('torch.FloatTensor')
    inputs.to(device)
    logps = model(inputs)
    ps=torch.exp(logps)
    probs, classes = ps.topk(topk, dim=1)
    return probs,classes
