
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import json

def load_checkpoint(path):
    checkpoint=torch.load(path+'/checkpoint.pth')
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(path_to_image):
    image=PIL.Image.open(path_to_image)
    im.thumbnail((256,256))
    np_image=np.array(im)

    np_image=np_image/255.0
    np_image=np_image-[0.485,0.456,0.406]
    np_image=np_image/[0.229,0.224,0.225]
    np_image=np_image.transpose((2,0,1))
    np_image=torch.tensor(np_image)
    return np_image

def predict(np_image, model, topk,device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device=='gpu' else "cpu")
    inputs=np_image.unsqueeze(0)
    inputs=inputs.type('torch.FloatTensor')
    inputs.to(device)
    logps = model(inputs)
    ps=torch.exp(logps)
    probs, classes = ps.topk(topk, dim=1)
    return probs,classes
