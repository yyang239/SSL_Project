import dnnlib
import legacy
import torch
import torchvision
from tqdm import tqdm
import pickle

with dnnlib.util.open_url('./network-snapshot-000088269.pkl') as f:
    ckpt = legacy.load_network_pkl(f)
    
D = ckpt['D'].Model
for x in D.MainLayers:
    x.DataType = torch.float32
D = D.to('cuda').requires_grad_(True)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)

train_label = []
train_grad = []

for x, y in tqdm(trainloader):
    x = 2 * (x - 0.5)
    x = x.to('cuda').requires_grad_(True)
    r, f = D(x)
    g, = torch.autograd.grad(outputs=r.sum(), inputs=f[-1], create_graph=True)
    train_label += [y]
    train_grad += [g.detach().cpu()]
    
train_label = torch.cat(train_label, dim=0)
train_grad = torch.cat(train_grad, dim=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

test_label = []
test_grad = []

for x, y in tqdm(testloader):
    x = 2 * (x - 0.5)
    x = x.to('cuda').requires_grad_(True)
    r, f = D(x)
    g, = torch.autograd.grad(outputs=r.sum(), inputs=f[-1], create_graph=True)
    test_label += [y]
    test_grad += [g.detach().cpu()]

test_label = torch.cat(test_label, dim=0)
test_grad = torch.cat(test_grad, dim=0)

torch.save(dict(label=train_label, grad=train_grad), 'train.pt')
torch.save(dict(label=test_label, grad=test_grad), 'test.pt')