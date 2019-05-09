import torch
from torchvision import transforms, datasets,models

def data_trans(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(224),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation([-30,90]),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    train = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test = datasets.ImageFolder(test_dir, transform=test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, sampler=None)
    valid_loader = torch.utils.data.DataLoader(valid,batch_size=64)
    test_loader = torch.utils.data.DataLoader(test,batch_size=128)
    return train_dir,valid_dir,test_dir,train_loader,valid_loader,test_loader
