## VGG: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 
## VGG pytorch实现
import torch.nn as nn
import torch
import requests
import os
# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

configs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(model_name: str):
    layers = []
    in_channels = 3
    assert model_name in ['vgg11', 'vgg13', 'vgg16', 'vgg19'], "Warning: model_name {} not in configs dict!".format(model_name)
    cfg = configs[model_name]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, model_name, cls_lengths = [4096, 4096], classes_num = 1000, init_weight=False):
        super(VGG, self).__init__()
        self.features = make_features(model_name)
        output_size = 512*7*7
        cls_lengths = [output_size] + cls_lengths
        self.classifier = nn.Sequential(
            *([module for i in range(len(cls_lengths)-1) \
                for module in (nn.Linear(cls_lengths[i], cls_lengths[i+1]), nn.ReLU(True), nn.Dropout())] + [nn.Linear(cls_lengths[-1], classes_num)]))
        if init_weight:
            self._initialize_weight()
    
    def forward(self, x):
        # x: (N, 3, 224, 224)
        x = self.features(x) # (N, 512, 7, 7)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                
def vgg(model_name="vgg16", classes_num = 1000, cls_lengths = [4096, 4096], pretrained = False):
    """
    input_size: (3, 224, 224)
    output_size: classes_num
    """
    if pretrained:
        assert classes_num == 1000, "WARNING: classes_num must be 1000 or default!"
        model = VGG(model_name=model_name)
        r = requests.get(model_urls[model_name]) 
        if not os.path.exists('vgg_pths'):
            os.makedirs('vgg_pths')
        save_path = "./vgg_pths/"+model_name+".pth"
        with open(save_path, "wb") as f:
            f.write(r.content)
        model.load_state_dict(torch.load(save_path))
        return model
    return VGG(model_name=model_name, classes_num=classes_num, cls_lengths = cls_lengths, init_weight=True)
        


