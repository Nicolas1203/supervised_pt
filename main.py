"""Main script for training a pytorch model in a supervised way.

TODO:
    * implement testing script only
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
import copy
import argparse
import sys

from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image
from skimage import io
from src.train import train
from src.utils.tensorboard import get_writer
from src.utils.data import get_loaders
from src.utils.visualization import plot_confusion_matrix 


parser = argparse.ArgumentParser(description="PyTorch supervised training code, with tensorboard visualisation")
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test-only', action='store_true', help='test only')   
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--nb-classes', '-nc', type=int, default=1695,
                    help='Number of classes for last layer')
parser.add_argument('--classes', '-c', default='category', help='Classes to consider as ground truth: category or channel.',
                    choices=['category, channel'])
parser.add_argument('--tensorboard', '-tb', help="Name of the tensorboard graph.",
                    default='')
parser.add_argument('--data-root-dir', default='/data/influencers/v1/',
                    help='Root dir containing the dataset to train on.')
parser.add_argument('--annotations', '-a', default='annotations_ch.csv', 
                    help="name of the csv containing data annotations.")
parser.add_argument('--resume', '-r', default='',
                    help="Resume old training. Load model given as resume parameter")
parser.add_argument('--train', '-t', action="store_true", help="Run the code in training mode.")
parser.add_argument('--confusion-matrix', '-cm', default='',
                    help="Create and save the confusion matrix.")
args =parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Model definition
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nb_classes)
    if len(args.resume) > 1:
        print(f"Resuming model {args.resume}...")
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Dataloaders and dataset sizes
    dataloaders, dataset_sizes = get_loaders(args.data_root_dir, args.annotations, args.batch_size)

    # tensorboard writer
    writer = get_writer(args.tensorboard)
    
    # Training
    if args.train:
        best_model = train(
            model,
            criterion,
            optimizer,
            scheduler,
            dataloaders,
            dataset_sizes,
            writer,
            start_epoch=args.start_epoch,
            end_epoch=args.start_epoch + args.epochs
            )
        writer.flush()
        sys.exit(0)
    if len(args.confusion_matrix) > 1:
        plot_confusion_matrix(
            model,
            dataloaders['val'],
            os.path.join(args.data_root_dir, args.annotations),
            classes_name=args.classes,
            fig_name=args.confusion_matrix
            )
        sys.exit(0)
    

if __name__ == '__main__':
    main()