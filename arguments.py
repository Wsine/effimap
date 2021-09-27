import argparse


devices = ['cpu', 'cuda']
datasets = ['cifar10', 'cifar100']
models = ['resnet32']


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--device', default='cuda', choices=devices)
parser.add_argument('--gpu', type=int, default=3, choices=(0, 1, 2, 3))
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-m', '--model', type=str, default='resnet32', choices=models)
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=datasets)
parser.add_argument('--eval', action='store_true', help='whether to evaluate the trained model only')

