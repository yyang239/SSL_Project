import dnnlib
import legacy
import torch
import torchvision
from tqdm import tqdm
import pickle
import math

train_data = torch.load('train.pt')
train_grad = train_data['grad']
train_label = train_data['label']

test_data = torch.load('test.pt')
test_grad = test_data['grad']
test_label = test_data['label']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grad, label):
        self.grad = grad
        self.label = label
    
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        return self.grad[idx], self.label[idx]
    
trainset = Dataset(train_grad, train_label)
testset = Dataset(test_grad, test_label)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

###############################

import torch.nn as nn
from BaselineGAN.FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class Convolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1):
        super(Convolution, self).__init__()
        
        self.Layer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Groups, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        return nn.functional.conv2d(x, self.Layer.weight.to(x.dtype), padding=self.Layer.padding, groups=self.Layer.groups)

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize):
        super(ResidualBlock, self).__init__()
        
        ExpandedChannels = InputChannels * ExpansionFactor

        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=1)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, KernelSize=KernelSize, Groups=Cardinality, ActivationGain=1)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class LinearHead(nn.Module):
    def __init__(self, InputChannels, SpatialSize, NumClasses):
        super(LinearHead, self).__init__()
        
        # self.res1 = ResidualBlock(768, 96, 2, 3)
        # self.res2 = ResidualBlock(768, 96, 2, 3)
        
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=SpatialSize, stride=1, padding=0, groups=InputChannels, bias=False))
        
        
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, NumClasses, bias=False))
        
    def forward(self, x):
        x = normalize(x, dim=1)
        x = normalize(x, dim=[2,3])
        # x = self.res1(x)
        # x = self.res2(x)
        x = self.Basis(x).view(x.shape[0], -1)
        return self.LinearLayer(x)
    
classifier = LinearHead(768, 4, 10).to('cuda')

#############################

import torch.optim as optim

optimizer = optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss = nn.CrossEntropyLoss()

for epoch in range(100): #1000000
    for x, y in trainloader:
        classifier.requires_grad = True
        classifier.zero_grad()
        
        x = x.to('cuda')
        y = y.to('cuda')
        
        pred = classifier(x)
        err = loss(pred, y)
        err.backward()
        optimizer.step()
        
        print(err)
        
        
correct = 0
total = 0
        
for x, y in tqdm(testloader):
    x = x.to('cuda')
    y = y.to('cuda')
        
    pred = classifier(x)
    _, pred = torch.max(pred, 1)
        
    total += y.size(0)
    correct += pred.eq(y).sum().item()
    
print(correct/total)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        