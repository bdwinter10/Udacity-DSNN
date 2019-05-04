import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets,models
import PIL
import numpy as np
import json

def building_and_training(arch,hidden_units,epochs,learning_rate,train_loader,valid_loader,device):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device=='gpu' else "cpu")
    model=arch(pretrained=True)
    for para in model.parameters():
        para.requires_grad=False
    classifier=nn.Sequential(nn.Linear(2048, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256,102),
                             nn.LogSoftmax(dim=1))
    model.fc=classifier
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.fc.parameters(), lr=learning_rate)
    model.to(device)
    #actual training
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train() #turn model back into training mode
    return model
    model.to('cpu')

# TODO: Do validation on the test set
def testing_model(model,test_loader):
    test_loss = 0
    accuracy_sum = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_sum += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()

        print(f"Test accuracy: {accuracy_sum/len(test_loader):.3f}")

# TODO: Save the checkpoint
def save_checkpoint(model,train_dir,save_dir):
    model.eval()
    model.class_to_idx = datasets.ImageFolder(train_dir).class_to_idx
    checkpoint = {'input_size': 2048,
                  'output_size': 102,
                  'state_dict': model.state_dict()}
    checkpoint.to('cpu')
    torch.save(checkpoint, save_dir+'/checkpoint.pth')

def load_checkpoint(save_dir):
    checkpoint=torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'])
    model.load_state_dict(checkpoint['state_dict'])

    return model
