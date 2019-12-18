import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

def get_min_max(list_probs):
    new_list = []
    for prob in list_probs:
        if prob == 0:
            continue
        new_list.append(prob)
    return (min(new_list), max(new_list))

class ImageCLEFWikipediaDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split):
        """
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.split = split.lower()

        assert self.split in {'train', 'test'}

        self.images = []
        self.probabilities = []

        # Read data files
        with open('training_labels40.json', 'r') as j:
            data = json.load(j)

            for path in data:
                self.images.append(path)
                self.probabilities.append(data[path])

        dataset_length = len(self.images)
        trainset_length = int(0.7 * dataset_length)
        valset_length = dataset_length - trainset_length

        if self.split == 'train':
            self.images = self.images[:trainset_length]
            self.probabilities = self.probabilities[:trainset_length]
        else:
            self.images = self.images[trainset_length:]
            self.probabilities = self.probabilities[trainset_length:]

        assert len(self.images) == len(self.probabilities)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read probabilities of this image
        probabilities = torch.tensor(self.probabilities[i])

        # Apply transformations
        image = transform(image, split=self.split)

        return image, probabilities

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        probabilities = list()

        for b in batch:
            images.append(b[0])
            probabilities.append(b[1])

        images = torch.stack(images, dim=0)
        probabilities = torch.stack(probabilities, dim=0)

        return images, probabilities  # tensor (N, 3, 224, 224), 1 list of N tensors (40 items) each

    def range(self):
        min_max = [get_min_max(probs) for probs in self.probabilities]
        return (min([range_probs[0] for range_probs in min_max]), max([range_probs[1] for range_probs in min_max]))
