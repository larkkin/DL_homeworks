import torch.nn as nn
import math



__all__ = ['ResNeXt', 'resnext']





class ResNextBlock(nn.Module):
    expansion = 1
    path_planes = 4


    def __init__(self, ext_planes, cardinality):
        super(ResNextBlock, self).__init__()

        planes = ResNextBlock.path_planes * cardinality 

        self.cardinality = cardinality
        self.relu = nn.ReLU(inplace = True)
        self.input_conv = nn.Conv2d(ext_planes, planes, kernel_size=1, stride=1, padding=0)
        self.input_bn = nn.BatchNorm2d(planes)
        self.transform_conv = nn.Conv2d(planes, planes,
                                        kernel_size=3, stride=1, padding=1, groups=cardinality)
        self.transform_bn = nn.BatchNorm2d(planes)
        self.output_conv = nn.Conv2d(planes, ext_planes, kernel_size=1, stride=1, padding=0)
        self.output_bn = nn.BatchNorm2d(ext_planes)
        
    def forward(self, residual):

        x = self.input_conv(residual)
        x = self.input_bn(x)
        x = self.relu(x)

        x = self.transform_conv(x)
        x = self.transform_bn(x)
        x = self.relu(x)
        
        x = self.output_conv(x)
        x = self.output_bn(x)
        x += residual
        x = self.relu(x)

        return x




class ResNext(nn.Module):

    def __init__(self, num_classes, in_planes, total_area, cardinality=32, depth=5, ext_planes=256):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of ResNeXt blocks.
            num_classes: number of classes
            ext_planes: base number of channels in each group.
            path_planes: number of channels within each path
        """
        super(ResNext, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.ext_planes = ext_planes
        self.num_classes = num_classes
        self.total_area = total_area

        self.conv1 = nn.Conv2d(in_planes, self.ext_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ext_planes)
        self.relu = nn.ReLU(inplace=True)
        self.main_blocks = nn.Sequential(*[ResNextBlock(self.ext_planes, self.cardinality) for i in range(self.depth)])            
        self.fc = nn.Linear(self.ext_planes * self.total_area, self.num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    
    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu(x)
        x = self.main_blocks.forward(x)
        x = x.view(-1, self.ext_planes * self.total_area)
        x = self.fc(x)
        return x





def resnext(pretrained_path=None, **kwargs):
    model = ResNext(**kwargs)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    return model