import os

import torch
import torch.nn as nn
import torch.nn.functional as F

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.rstrip('/').split('/')[:-1])

################################################################################
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model_map = {'baseline': Baseline}
################################################################################

def load_model(model, checkpoint=None):
    if model == 'baseline':
        model =  model_map[model]()
    else:
        raise NotImplementedError

    if checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(dir_path, 'untracked', 'model', checkpoint)))
    
    return model