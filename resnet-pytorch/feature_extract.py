# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

__editor__ = "Il-Ji Choi, Vuno. Inc."  # test file
__editor_email__ = "choiilji@gmail.com"

import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.model import CifarResNeXt
import numpy as np
import time


def get_args():
    parser = argparse.ArgumentParser(description='Test ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=10)
    # Checkpoints
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    parser.add_argument("--train_set", type=int, default=0, help='phase of dataset')
    parser.add_argument("savename", type=str, default="result", help="savename of result")
    args = parser.parse_args()
    return args


def test():
    # define default variables
    args = get_args()  # divide args part and call it as function
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    state = {k: v for k, v in args._get_kwargs()}

    # prepare test data parts
    if args.train_set == 0:
        phase = False
    else:
        phase = True

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_data = dset.CIFAR100(args.data_path, train=phase, transform=test_transform, download=True)
    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10(args.data_path, train=phase, transform=test_transform, download=True)
        nlabels = 10
    else:
        test_data = dset.CIFAR100(args.data_path, train=phase, transform=test_transform, download=True)
        nlabels = 100

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)


    # initialize model and load from checkpoint
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.widen_factor)
    loaded_state_dict = torch.load(args.load, map_location='cpu')
    temp = {}
    for key, val in list(loaded_state_dict.iteritems()):
        # parsing keys for ignoring 'module.' in keys
        temp[key[7:]] = val
    loaded_state_dict = temp
    net.load_state_dict(loaded_state_dict)

    # paralleize model
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if args.ngpu > 0:
        net.cuda()

    # use network for evaluation
    net.eval()

    # calculation part
    start = time.clock()
    result = np.zeros((len(test_data), net.stages[3]), dtype=np.float32)
    result_target = np.zeros(len(test_data))

    print(result.shape)
    for batch_idx, (data, target) in enumerate(test_loader):
        result_target[batch_idx * args.test_bs:(batch_idx + 1) * args.test_bs] = target.numpy()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        # forward
        output = net(data).detach().numpy()
        result[batch_idx*args.test_bs:(batch_idx+1)*args.test_bs,:] = output
        print("batch: {0}, average_time: {1}s, time_left:{2}s".format(batch_idx, (time.clock()-start)/(batch_idx+1), (time.clock()-start)/(batch_idx+1)*(len(test_loader)-batch_idx-1)))
        if batch_idx % 1000==0:
            np.save(args.savename+".npy",result)
    np.save(args.savename, result)
    np.save(args.savename+"_target", result_target)

if __name__ == '__main__':
    test()
