import os

import torchvision
import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.rstrip('/').split('/')[:-1])

################################################################################
def CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(dir_path, 'untracked/data'), train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(dir_path, 'untracked/data'), train=False,
                                           download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, testset, classes

dataset_map = {'CIFAR10': CIFAR10}
################################################################################

def load_dataset(dataset):
    if dataset in dataset_map:
        return dataset_map[dataset]()
    else:
        raise NotImplementedError