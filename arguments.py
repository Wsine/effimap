import argparse


devices = ('cpu', 'cuda')
datasets = ('cifar100', 'tinyimagenet')
models = ('resnet32', 'resnet18')
tasks = ('clf', 'reg')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--device', default='cuda', choices=devices)
parser.add_argument('--gpu', type=int, default=3, choices=(0, 1, 2, 3))
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-m', '--model', type=str, default='resnet32', choices=models)
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=datasets)
parser.add_argument('-e', '--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--fuzz_energy', type=int, default=10)
parser.add_argument('--num_model_mutants', type=int, default=100)
parser.add_argument('--num_input_mutants', type=int, default=200)
parser.add_argument('--task', type=str, default='clf', choices=tasks)

