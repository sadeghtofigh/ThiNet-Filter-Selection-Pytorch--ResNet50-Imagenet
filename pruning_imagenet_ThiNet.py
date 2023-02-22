# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str, timing
from typing import Dict, Iterable, Callable
# from models import print_log
import models
import random
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
import Filter_index

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    default='',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='C:/Users/...',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')

# compress rate
parser.add_argument('--rate_thinet', type=float, default=0.2, help='The reducing ratio of pruning')
parser.add_argument('--random_data', default=10000, type=int, metavar='images', help='The number of random images for filter selection by thinet')
parser.add_argument('--random_entry', default=10, type=int, metavar='entry', help='number of the random instances in the output')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')


args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()


def main():
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')
    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=args.use_pretrain)
    if args.use_sparse:
        model = import_sparse(model)
    #print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Thinet pruning rate: {}".format(args.rate_thinet), log)
    print_log("Number of random images for pruning : {}".format(args.random_data), log)
    print_log("Number of random instances in the output: {}".format(args.random_entry), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)
    print_log("Workers         : {}".format(args.workers), log)

    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Random indices for the filter selectionj sub-dataset
    data_batch_thinet = np.random.randint(1281000, size=int(args.random_data))

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, sampler=data_batch_thinet, pin_memory=True)
    print_log("Length of dataloader:   {}".format(len(train_loader)), log)


    for p_l in Filter_index.MI:
        print_log("------------------- Filter selection of layer: {} -------------------".format(p_l), log)
        m = Mask(model,train_loader, p_l, log)
        m.init_length()
        m.init_mask(args.rate_thinet)


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()
    

class Mask:
    def __init__(self, model, train_loader, p_layer, log):
        self.model_size = {}
        self.model_length = {}
        self.thinet_rate = {}
        self.mat = {}
        self.model = model
        self.train_loader = train_loader
        self.mask_index = []
        self.random_data = args.random_data
        self.random_entry = args.random_entry
        self.p_layer = p_layer
        self.next_layer_name = Filter_index.next_layer_ind[self.p_layer]
        self.log = log
        self.output_extract = EntrySum(Filter_index.layer_Shape[self.next_layer_name+".weight"], self.log)

    def get_thinet_codebook(self, weight_torch, compress_rate, length):
        T_set = []
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(
                weight_torch.size()[0] * (compress_rate))  # number of filters must be pruned in this layer
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            CO = 0
            start_time = time.time()
            while len(T_set) < filter_pruned_num:
                filter_T_norm = {}
                for fil in range(weight_torch.size()[0]):
                    fs_codebook = np.ones(length)
                    if fil in T_set:
                        continue
                    for i in range(weight_torch.size()[0]):
                        if i not in T_set and i != fil:
                            fs_codebook[i * kernel_length: (i + 1) * kernel_length] = 0
                    filter_T_norm[fil] = round(self.ThiNEt_tra(self.convert2tensor(fs_codebook)), 2)
                elapsed_time = time.time() - start_time
                print_log("Iteration: {}, Output values = {}, Duration:{:.1f} minutes".format(CO+1, filter_T_norm,
                                                                                         elapsed_time/60), self.log)
                CO += 1
                T_set.append(self.mi_val_dic(filter_T_norm))
        return T_set

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()
            #print("The size of the {}th filter is {}".format(index,item.size()))
        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def mi_val_dic(self, di):
        min_value = float("inf")
        min_key = None
        for key, value in di.items():
            if value < min_value:
                min_value = value
                min_key = key
        return min_key

    def init_rate(self, rate_thinet_per_layer): 
            for index, item in enumerate(self.model.parameters()):
                self.thinet_rate[index] = 1
            self.thinet_rate[self.p_layer] = rate_thinet_per_layer
            # different setting for  different architecture
            if args.arch == 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [21, 36, 51]
            elif args.arch == 'resnet34':
                last_index = 108
                skip_list = [27, 54, 93]
            elif args.arch == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
            elif args.arch == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
            elif args.arch == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
            self.mask_index = [x for x in range(0, last_index, 3)]
            # skip downsample layer
            if args.skip_downsample == 1:
                for x in skip_list:
                    self.thinet_rate[x] = 1
                    self.mask_index.remove(x)
                print("The mask index is:----->", self.mask_index)
            else:
                pass

    def init_mask(self, rate_thinet_per_layer):
        self.init_rate(rate_thinet_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index == self.p_layer:
                print_log("Layer: {}, TSet: {} ".format(index,
                          self.get_thinet_codebook(item.data, self.thinet_rate[index], self.model_length[index])), self.log)

    def ThiNEt_tra(self, p_codebook):
        for index, item in enumerate(self.model.parameters()):
            if index == self.p_layer:
                origin_layer = item.data
                a = item.data.view(self.model_length[index]).cpu()
                b = a * p_codebook
                item.data = b.view(self.model_size[index]).cuda()
        norm = 0
        resnet_features = FeatureExtractor(self.model, layers=[self.next_layer_name])
        for i, (m_input, _) in enumerate(self.train_loader):
            input_var = torch.autograd.Variable(m_input)
            features = resnet_features(input_var)
            for m_item, m_output in features.items():
                with torch.no_grad():
                    norm += self.output_extract.sum_cal(m_output)
        for index, item in enumerate(self.model.parameters()):
            if index == self.p_layer:
                item.data = origin_layer.cuda()
        return norm/len(self.train_loader)

class FeatureExtractor(nn.Module):
        def __init__(self, model: nn.Module, layers: Iterable[str]):
            super().__init__()
            self.model = model
            self.layers = layers
            self._features = {layer: torch.empty(0) for layer in layers}

            for layer_id in layers:
                layer = dict([*self.model.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))

        def save_outputs_hook(self, layer_id: str) -> Callable:
            def fn(_, __, output):
                self._features[layer_id] = output

            return fn

        def forward(self, x):
            _ = self.model(x)
            return self._features


class EntrySum:
    def __init__(self, tens_shape, log):
        self.tens_shape = tens_shape
        self.random_channel_indices = np.random.randint(0, self.tens_shape[1], args.random_entry)
        self.random_height_indices = np.random.randint(0, self.tens_shape[2], args.random_entry)
        self.random_width_indices = np.random.randint(0, self.tens_shape[3], args.random_entry)
        self.log = log
        print_log("Channel index:{}\nHeight index:{}\nWidth index:{}"
                  .format(self.random_channel_indices, self.random_height_indices, self.random_width_indices), log)

    def sum_cal(self, inp):
        selected_output = inp[:, self.random_channel_indices, self.random_height_indices,
                              self.random_width_indices]
        sum_of_squares = torch.sum(selected_output ** 2)
        return float(sum_of_squares)


if __name__ == '__main__':
    main()
