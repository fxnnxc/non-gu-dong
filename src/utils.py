import matplotlib.pyplot as plt
import numpy as np

import torchvision

# functions to show an image
def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()