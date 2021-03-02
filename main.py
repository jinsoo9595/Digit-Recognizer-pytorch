import os
import time
import shutil
import parser
import argparse

import torch
import torch.nn
import torch.optim
# import torchvision.models as models

# import Models.resnet as resnet
# import Models.efficient as efficient

from utils import prepare_dataloaders
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):

    """ Settings for training """
    torch.manual_seed(args.seed)

    """ Settings for HW """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.device_count() > 1

    """ Prepare the data loaders """
    train_loader, valid_loader, train_size, valid_size = prepare_dataloaders(args)

    """ Load the training model """
    # if args.arch == "efficient-l2":
    #     model = efficient.efficientL2(num_classes=args.num_classes)
    # elif args.arch == 'resnet18':
    #     model = resnet.resnet18(num_classes=args.num_classes)
    # elif args.arch == 'resnet34':
    #     model = resnet.resnet34(num_classes=args.num_classes)
    # elif args.arch == 'resnet50':
    #     model = resnet.resnet50(num_classes=args.num_classes)
    # elif args.arch == 'resnet101':
    #     model = resnet.resnet101(num_classes=args.num_classes)
    # elif args.arch == 'resnet152':
    #     model = resnet.resnet152(num_classes=args.num_classes)

    # """ Set specified HW """
    # model = torch.nn.DataParallel(model.to(device))
    #
    # """ define loss function. """
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    #
    # """ define optimizer """
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    # print('[Info] Total parameters {} '.format(count_parameters(model)))
    #
    # if args.resume or args.test:
    #     print('[Info] Loading checkpoint.')
    #     checkpoint = load_checkpoint(args.save)
    #     arch = checkpoint['arch']
    #     args.epoch = checkpoint['epoch']
    #     state_dict = checkpoint['state_dict']
    #     model.load_state_dict(state_dict)
    #     print('[Info] epoch {} arch {}'.format(args.epoch, arch))
    #
    # """ run evaluate """
    # if args.evaluate:
    #     _ = run_epoch(model, 'valid', [args.epoch, args.epoch], criterion, optimizer, valid_loader, valid_size, device)
    #     return
    #
    # """ run train """
    # best_acc1 = 0.
    # for e in range(args.epoch, args.n_epochs + 1):
    #     adjust_learning_rate(optimizer, e, args)
    #
    #     """ train for one epoch """
    #     _ = run_epoch(model, 'train', [e, args.n_epochs], criterion, optimizer, train_loader, train_size, device)
    #
    #     """ evaluate on validation set """
    #     with torch.no_grad():
    #         acc1 = run_epoch(model, 'valid', [e, args.n_epochs], criterion, optimizer, valid_loader, valid_size, device)
    #
    #     # Save checkpoint.
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)
    #     save_checkpoint({
    #         'epoch': e,
    #         'arch': args.arch,
    #         'state_dict': model.module.state_dict(),
    #         'best_acc1': best_acc1,
    #         'optimizer': optimizer.state_dict(),
    #     }, is_best, args.save_path)
    #     print('[Info] acc1 {} best@acc1 {}'.format(acc1, best_acc1))

def run_epoch(model, mode, epoch, criterion, optimizer, data_loader, dataset_size, device):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    tq = tqdm(data_loader, desc='  - (' + mode + ')   ', leave=False)
    for data, target in tq:
        # prepare data
        data, target = data.to(device), target.to(device)

        # forward
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        if mode == 'train':
            # compte gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tq.set_description(' - ({}) [ epoch: {}/{} loss: {:.3f}/{:.3f} ] '.format(mode, epoch[0], epoch[1], losses.val, losses.avg))
        #tqdm.write
    tqdm.write(' - ({})  [ epoch: {}\ttop@1: {:.3f}\ttop@5: {:.3f}\tloss: {:.3f}\ttime: {:.3f}]'.format(mode, epoch, top1.avg, top5.avg, losses.avg, (time.time() - start)/60.))
    return top1.avg

def save_checkpoint(state, is_best, prefix):
    filename='checkpoints/{}_checkpoint.chkpt'.format(prefix)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/{}_best.chkpt'.format(prefix))
    print(' - [Info] The checkpoint file has been updated.')

def load_checkpoint(prefix):
    filename='checkpoints/{}_checkpoint.chkpt'.format(prefix)
    return torch.load(filename)

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        bsz = target.size(0)
        '''
            https://pytorch.org/docs/stable/torch.html#torch.topk
            torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        '''
        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / bsz))
        return res

if __name__ == '__main__':
    ''' Main function '''
    parser = argparse.ArgumentParser(description='Implement Digit recognizer on MNIST datset using pytorch')

    parser.add_argument('--mode', type=str, default='train',
                        help='train/evelaute/test')
    parser.add_argument('--arch', type=str, default='efficient-l2',
                        help='classification model (defulat: efficient)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='B',
                        help='input batch size for training (defulat: 128)')
    parser.add_argument('--resume', type=bool, default=False, metavar='R',
                        help='resume the model from epoch to epochs (defulat: 1)')
    parser.add_argument('--epoch', type=int, default=1,
                        help='number of start epoch to train (defulat: 1)')
    parser.add_argument('--epochs', type=int, default=512, metavar='E',
                        help='number of epochs to train (defulat: 512)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (defulat: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='momentum (defulat: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='LR',
                        help='weight decay (defulat: 5e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='training dataset.  (mnist)')
    parser.add_argument('--num_classes', type=int, default='10',
                        help='number of class for training dataset.  (mnist)')
    parser.add_argument('--data_path', type=str, default='./Datas',
                        help='Path of datasets (default: ./Datas)')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Path of datasets (default: ./checkpoints)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of workers (default: 4)')

    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    main(args)
