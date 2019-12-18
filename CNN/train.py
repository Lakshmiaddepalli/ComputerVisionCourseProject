import multiprocessing
import os
import time
import torch.backends.cudnn as cudnn
import torch.nn
import torch.optim
import torch.utils.data

from datasets import ImageCLEFWikipediaDataset
from model import TextTopicNetCNN
from utils import *


data_dir = './'

# Model parameters
n_topics = 40  # number of topics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint_filename = './checkpoint_ssl.pth.tar'
checkpoint = checkpoint_filename if os.path.exists(checkpoint_filename) else None # path to model checkpoint, None if none
batch_size = 64  # batch size
start_epoch = 0  # start at this epoch
epochs = 400  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_loss = 100.  # assume a high loss at first
workers = multiprocessing.cpu_count() - 1 # 4 # number of workers for loading data in the DataLoader
print_freq = 200  # print training or validation status every __ batches
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

def CXE(predicted, target):
    return -(target * torch.log(predicted)).sum(dim=1).mean()

def main():
    """
    Training and validation.
    """
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint, lr

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = TextTopicNetCNN(n_topics)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        # biases = list()
        # not_biases = list()
        # for param_name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if param_name.endswith('.bias'):
        #             biases.append(param)
        #         else:
        #             not_biases.append(param)

        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        # lr = lr/10
        # optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, weight_decay=weight_decay)

        # optimizer = torch.optim.Adam(params=model.parameters())
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.BCELoss().to(device)
    # criterion = torch.nn.MSELoss().to(device)
    criterion = CXE

    # Custom dataloaders
    train_dataset = ImageCLEFWikipediaDataset('train')
    val_dataset = ImageCLEFWikipediaDataset('test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)
    
    epoch_time = AverageMeter()  # epoch time
    epoch_loss = AverageMeter()  # epoch loss
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)

        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop

        start = time.time()

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        epoch_time.update(time.time() - start)
        epoch_loss.update(val_loss)

        print('Epoch: [{0}/{1}]\t'
              'Epoch Time {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, epochs,
                                                              epoch_time=epoch_time,
                                                              loss=epoch_loss))

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)

        start = time.time()


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, probabilities) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 224, 224)
        probabilities = probabilities.to(device)

        # Forward prop.
        predicted_probabilities = model(images)  # (N, n_topics)

        # probabilities *= 100
        # predicted_probabilities *= 100

        # Loss
        loss = criterion(predicted_probabilities, probabilities)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            print(predicted_probabilities[0])
            print(probabilities[0])
    del images, probabilities  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, probabilities) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 224, 224)
            probabilities = probabilities.to(device)

            # Forward prop.
            predicted_probabilities = model(images)  # (N, n_topics)

            # probabilities *= 100
            # predicted_probabilities *= 100

            # Loss
            loss = criterion(predicted_probabilities, probabilities)  # scalar

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':
    main()
