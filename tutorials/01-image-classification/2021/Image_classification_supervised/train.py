import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import CNN
from dataset import cats_dogs_dataset

from tqdm import tqdm
import json
import argparse

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def validate(model, val_loader):
    
    '''
    validation loop
    - check intermediate model performance
    - prints accuracy, returns accuracy
    '''
    
    val_loop = tqdm(val_loader, total = len(val_loader), leave = True)

    num_correct = 0
    num_samples = 0
    
    # set the model for validation, gradients are *NOT* updated
    model.eval()

    # yet another way to prevent update of gradients
    with torch.no_grad():
        
        for x, y in val_loop:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
    print(f'Got {num_correct} / {num_samples} with accuracy\
          {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()
    
    # save model
    PATH = './dogs_cats_binary.pth'
    torch.save(model.state_dict(), PATH)
    print('The model is saved!')
    
    return f'{float(num_correct)/float(num_samples)*100:.2f}'

    
    
def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs):
    
    '''
    training loop
    '''
    
    # PyTorch way to set the model for training (with the gradient updates)
    model.train()
    
    # train for N epochs
    for epoch in range(num_epochs):
        
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        
        # perform validation every 2 epochs
        if epoch % 2 == 0 and epoch != 0:
            val_acc = validate(model, validation_loader)
            
        for imgs, labels in loop:
            
            imgs = imgs.to(device)
            labels = labels.to(device)
                        
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())
            
    print('Finished training!')



def main(args):
    
    '''
    main function
    - describes the whole model pipeline
    - it should refer to a separate train function, validation function
    - loads datasets, initializes models and hyperparameters
    - sets learning rate and optimizer, passes arguments to the training loop
    '''
    
    # load my hyperparameters
    with open(args.config_file, 'r') as f:
        hyps = json.load(f)

    # why do we need to transform images?
    # - images differ from each other size-wise
    # - random cropping can be helpful because it allows your model to learn different
    # ways your object can be represented
    # - normalisation helps to speed up training / utilise pre-trained models better
    transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                #transforms.RandomCrop((x, x)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    
    # load the dataset
    dataset = cats_dogs_dataset(args.data_dir, args.annotations_file, transform=transform)
    
    # split dataset into training and validation (25k : 20k + 5k)
    train_set, validation_set = torch.utils.data.random_split(dataset, [20000 , 5000])
    
    # train loader
    train_loader = DataLoader(dataset=train_set, shuffle=bool(hyps['shuffle']),
                                   batch_size=hyps['batch_size'], num_workers=hyps['num_workers'],
                                   pin_memory=bool(hyps['pin_memory']))    
    # val loader
    validation_loader = DataLoader(dataset=validation_set, shuffle=bool(hyps['shuffle']),
                                   batch_size=hyps['batch_size'], num_workers=hyps['num_workers'],
                                   pin_memory=bool(hyps['pin_memory']))    
    
    
    # initialize and import model to GPU
    model = CNN().to(device)
    #print(model)
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # define loss (Binary Cross-Entropy for the binary classification task)
    criterion = nn.BCELoss()
    # define optimizer (Adam) on model parameters with the specified learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=hyps['learning_rate'])
    
    train(model, criterion, optimizer, train_loader, validation_loader, hyps['num_epochs'])



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config.json', help='config file with hyperparameters')
    parser.add_argument('--data_dir', type=str, default='./data/train/', help='directory with the data for training')
    parser.add_argument('--annotations_file', type=str, default='./train_df.csv', help='file with annotations')
    arguments = parser.parse_args()
    
    main(arguments)