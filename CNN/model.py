from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class CNN(nn.Module):
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Probability(nn.Module):
    r"""Applies the Probability function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Probability is defined as:

    .. math::
        \text{Probability}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Probability will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Probability and itself.
        Use `Probability` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Probability(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']

    def __init__(self, dim=1):
        super(Probability, self).__init__()
        self.dim = dim
        

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        input = F.relu(input)
        return input / torch.sum(input, dim=self.dim).view(-1, 1)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

class TextTopicNetCNN(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self, n_topics=40):
        super(TextTopicNetCNN, self).__init__()

        # self.cnn = torchvision.models.vgg16(pretrained=False, num_classes=n_topics)
        self.cnn = torchvision.models.alexnet(pretrained=False, num_classes=n_topics)
        # self.cnn = AlexNet(num_classes=n_topics)
        # self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.probability = Probability()

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 224, 224)
        :return: class predictions
        """

        # TextTopicNetCNN outputs
        # return self.softmax(self.sigmoid(self.cnn(image)))
        # return self.probability(self.cnn(image))
        return self.softmax(self.cnn(image))
        # if not self.training:
        #     return self.probability(self.cnn(image))
        # else:
            # return self.softmax(self.cnn(image))

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        cnn_state_dict = self.cnn.state_dict()
        cnn_param_names = list(cnn_state_dict.keys())

        # Pretrained VGG base
        # pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(cnn_param_names[:-2]):  # excluding fc2 parameters
            # print(pretrained_param_names[i])
            # cnn_state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
            if param in pretrained_state_dict:
                cnn_state_dict[param] = pretrained_state_dict[param]

        # # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # # ...operating on the 2D image of size (C, H, W) without padding

        self.cnn.load_state_dict(cnn_state_dict)

        print("\nLoaded base model.\n")
