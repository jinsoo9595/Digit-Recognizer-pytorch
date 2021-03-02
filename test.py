import os
import time
import shutil
import parser
import argparse

import torch
import torch.nn
import torch.optim
import torchvision.models as models

import Models.resnet as resnet
import Models.efficient as efficient

from utils import prepare_dataloaders
from tqdm import tqdm

def main(args):

    """ Settings for training """
    torch.manual_seed(args.seed)

    """ Settings for HW """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Prepare the data loaders """
    test_loader, test_size = prepare_test_dataloaders(args)

    """ Load the training model """
    if args.arch == "efficient-l2":
        model = efficient.efficientL2(num_classes=num_classes)
    elif args.arch == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes)
    elif args.arch == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes)
    elif args.arch == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes)
    elif args.arch == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes)
    elif args.arch == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes)

    if args.resume:
        print('[Info] Loading checkpoint.')
        checkpoint = load_checkpoint(args.save)
        arch = checkpoint['arch']
        args.epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print('[Info] epoch {} arch {}'.format(args.epoch, arch))
    run_test(model, test_loader, device)

def run_test(model, data_loader, device):
    model.eval()
    outputs = []
    tq = tqdm(data_loader, desc='  - (' + mode + ')   ', leave=False)
    for data, target in tq:
        data, target = data.to(device), target.to(device)
        output = model(data)
        outputs.append(outupt)
    print(outputs)

if __name__ == '__main__':
    ''' Main function '''
    parser = argparse.ArgumentParser(description='Implement Digit recognizer on MNIST datset using pytorch')

    parser.add_argument('--mode', type=str, default='train',
                        help='train/evelaute/test')
    parser.add_argument('--arch', type=str, default='efficient-l2',
                        help='classification model (defulat: efficient)')
    parser.add_argument('--resume', type=bool, default=False, metavar='R',
                        help='resume the model from epoch to epochs (defulat: 1)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='training dataset.  (mnist)')
    parser.add_argument('--num_classes', type=int, default='10',
                        help='number of class for training dataset.  (mnist)')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path of datasets (default: ./data)')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Path of datasets (default: ./checkpoints)')
    args = parser.parse_args()
    for arg in args:
        arg = arg.lower()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

	main(args)
