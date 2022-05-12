
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

################################################################################
### Inputs
# target_class: (int tensor) target class index.
# M: (int) the number of samples (interpolation, noise, ...).
# sigma: (int) standard deviation for gaussian noise.
################################################################################
### Outputs
# output: dictionary (output['attribution'] must be provided as aligned with the input x.)
################################################################################


def integrated_gradient(model, x, baseline, target_class, M, device, **temp):
    def make_interpolation(x, base, M):
        lst = [] 
        for i in range(M):
            alpha = i/M
            interpolated =x * (alpha) + base * (1-alpha)
            lst.append(interpolated.clone())
        return torch.stack(lst)

    X = make_interpolation(x, baseline, M)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()

    x = x.to(device)
    baseline = baseline.to(device)
    
    logits = model(X)

    index = torch.ones((logits.size()[0], 1), dtype=torch.int64) * target_class
    index = index.to(device)
    logits = logits.gather(1, index)

    model.zero_grad()
    logits.sum().backward()

    gradient = (x - baseline) * X.grad.sum(axis=0)

    output = {}
    output['attribution'] = gradient
    return output


def vanilla_gradient(model, x, target_class, device, **temp):
    X = x.unsqueeze(0)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    logits = model(X)
    
    index = torch.ones((logits.size()[0], 1), dtype=torch.int64) * target_class
    index = index.to(device)
    logits = logits.gather(1, index)

    model.zero_grad()
    logits.backward()

    gradient = X.grad[0]

    output = {}
    output['attribution'] = gradient
    return output


def smooth_gradient(model, x, target_class, M, sigma, device, **temp):
    def make_perturbation(x, M, sigma=1):
        lst = [] 
        for i in range(M):
            noise = torch.normal(0, sigma, size=x.size()).to(device)
            lst.append(x.clone() + noise.clone())
        return torch.stack(lst)
    
    X = make_perturbation(x, M, sigma)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    Y = torch.stack([target_class for i in range(len(X))]).to(device)
    
    logits = model(X)

    index = torch.ones((logits.size()[0], 1), dtype=torch.int64) * target_class
    index = index.to(device)
    logits = logits.gather(1, index)

    model.zero_grad()
    logits.sum().backward()

    gradient = X.grad.sum(axis=0)

    output = {}
    output['attribution'] = gradient
    return output

