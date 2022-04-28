from os import unlink
import unittest 
import torch 
import sys 
sys.path.append("../")

class Test(unittest.TestCase):

    def test_1(self):
        from src.class_activation_map import class_activation_map
        from src.class_activation_map import gradient_class_activation_map

        K = 32
        width, height = (64, 64)
        last_conv_feature = torch.rand(1, K, width, height)
        class_weights = torch.rand(K)

        cam = class_activation_map(last_conv_feature, class_weights)
        
        print("number of filters :", K)
        for k,v in cam.items(): 
            print(k, v.size())

        K = 32
        width, height = (64, 64)
        last_conv_feature = torch.rand(1, K, width, height)
        class_weights = torch.rand(K)

        cam = gradient_class_activation_map(last_conv_feature, class_weights)
        
        print("number of filters :", K)
        for k,v in cam.items():
            print(k, v.size())

