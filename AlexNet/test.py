import torch
from model import AlexNet
import os

def test():
    model = AlexNet(num_classes=5)
    model.load_state_dict(torch.load('checkpoints/alexnet_params.pth'))
    print(model)
    
if __name__ == '__main__':
    test()