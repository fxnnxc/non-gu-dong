
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


def integrated_gradient(model, x, x_baseline, target_class, M, **temp):
    output = {}

    return output


def vanilla_gradient(model, x, **temp):
    output = {}
    
    return output


def smooth_gradient(model, x, target_class, M, sigma, **temp):
    output = {}
    
    return output

