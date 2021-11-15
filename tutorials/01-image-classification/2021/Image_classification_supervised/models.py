import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from collections import OrderedDict


class CNN(nn.Module):
    '''
    construct CNN model for classification
    - consists of pre-trained ResNet model
    - the last layer is replaced with a linear layer for classification task
    '''
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()
        
        # this VGG16 is trained on ImageNet
        self.vgg16 = models.vgg16(pretrained=True)
        
        self.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.5)),
                                  ('fc2', nn.Linear(4096, 4096)),
                                  ('relu2', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.5)),
                                  ('fc3', nn.Linear(4096, num_classes)),
                                  ('output', nn.Sigmoid())
                                  ]))

        self.vgg16.classifier = self.classifier
        

    def forward(self, images):
        '''
        pass features through (i) the pre-trained model, (ii) activation functions
        result: 0 or 1, indicating one of the options for binary classification
        '''
        features = self.vgg16(images)
        return features.squeeze(1)