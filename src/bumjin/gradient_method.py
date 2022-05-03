
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def integrated_gradient(model, x, target_class, baseline, M, criterion=nn.CrossEntropyLoss(), device=torch.device('cuda'), **temp):

    def make_interpolation(x, base, M):
        lst = [] 
        for i in range(M):
            alpha = i/M
            interpolated =x * (alpha) + base * (1-alpha)
            lst.append(interpolated.clone())
        return torch.stack(lst)

    model.to(device)

    X = make_interpolation(x, baseline, M)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    x = x.to(device)
    baseline = baseline.to(device)
    
    Y = torch.stack([target_class for i in range(len(X))]).to(device)
    
    output = model.forward(X)
    loss = criterion(output, Y)
    loss.backward()

    gradient = X.grad

    IG = (x-baseline) * gradient.sum(axis=0)
    output = {}
    output['attribution'] = IG
    output['X'] = X
    output['gradient'] = gradient
    return output

def vanilla_gradient(model, x, target_class, criterion=nn.CrossEntropyLoss(), device=torch.device('cuda'), **temp):
    model.to(device)
    X = x.unsqueeze(0)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    Y = Variable(target_class).unsqueeze(0).to(device)
    
    output = model.forward(X)
    loss = criterion(output, Y)
    loss.backward()

    vanilla_gradient = X.grad[0]

    output = {}
    output['attribution'] = vanilla_gradient
    return output


def smooth_gradient(model, x, target_class, M, sigma, criterion=nn.CrossEntropyLoss(), device=torch.device('cuda'), **temp):
    def make_perturbation(x, M, sigma=1):
        lst = [] 
        for i in range(M):
            noise = torch.normal(0, sigma, size=x.size()).to(device)
            lst.append(x.clone() + noise.clone())
        return torch.stack(lst)
    
    model.to(device)
    X = make_perturbation(x, M, sigma)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    Y = torch.stack([target_class for i in range(len(X))]).to(device)
    
    output = model.forward(X)
    loss = criterion(output, Y)
    loss.backward()

    vanilla_gradient = X.grad.sum(axis=0)

    output = {}
    output['attribution'] = vanilla_gradient
    return output



    
