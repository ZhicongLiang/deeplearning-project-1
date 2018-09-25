
import argparse
import numpy as np
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='./model_best-cpu.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

def remove_module_in_keys(state_dict):
    '''
    since the checkpoint has 'module' in its state_dict's keys,
    which could not be recognised by model, so we need to remove this string
    from the keys
    '''
    new_state_dict = dict()
    for key in state_dict:
        if 'module' in key:
            new_state_dict[key.replace('.module', '')] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def load_data(path):
    # download data online
    # num_workers -- the number of processes to load in the data
    # if it is set as 0, that means you load the data in main process
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=path, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    
    # pretrained model: accuracy of 0.9215 in validation
    args = parser.parse_args()
    model = vgg.__dict__[args.arch]()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(remove_module_in_keys(checkpoint['state_dict']))
    
    path_data = 'C:/Users/ZhicongLiang/.keras/datasets/' 
    train_loader, val_loader= load_data(path_data)
    
    # get the bottle neck feature and switch to evaluate mode
    new_model = nn.Sequential(*list(model.children()))[0]
    new_model.eval()

    tic = time.time()
    
    feature = np.zeros((0,512))
    for i, (_input, target) in enumerate(val_loader):
        print(i,'/ 78')
        with torch.no_grad():
            input_var = torch.autograd.Variable(_input)
            target_var = torch.autograd.Variable(target)
        output = new_model(input_var)
        new_feature = np.array(output.data.reshape(output.size()[:2]))
        feature = np.append(feature,new_feature,axis=0)    
    pickle.dump(feature, open('vgg19_test_bottleneck_feature_cifar10.p','wb'))
    
    toc = time.time()
    print('Used time:', (toc-tic))