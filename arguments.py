import argparse

class ArgsWrapper(argparse.ArgumentParser):
    def add_dispatch(self, dp):
        self.add_argument(
            f'--{dp.attr}',
            type=str, required=True, choices=list(dp.registry.keys())
        )
        return self

    def parse_args(self, *args, **kwargs):
        opt = super().parse_args(*args, **kwargs)
        opt = self._get_num_classes(opt)
        return opt

    def _get_num_classes(self, opt):
        if opt.dataset in ('cifar10', 'mnist', 'svhn', 'stl10'):
            opt.num_classes = 10
        elif opt.dataset == 'cifar100':
            opt.num_classes = 100
        elif opt.dataset == 'tinyimagenet':
            opt.num_classes = 200
        else:
            raise ValueError('Invalid dataset name')
        return opt


devices = ['cpu', 'cuda']
datasets = ['cifar10', 'cifar100', 'mnist', 'svhn', 'stl10', 'tinyimagenet']
models = ['resnet32', 'mlp', 'svhn', 'stl10', 'resnet18', 'resnet20']


parser = ArgsWrapper()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--device', default='cuda', choices=devices)
parser.add_argument('--gpu', type=int, default=3, choices=(0, 1, 2, 3))
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-m', '--model', type=str, default='resnet32', choices=models)
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=datasets)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('--num_model_mutants', type=int, default=100)
parser.add_argument('--num_input_mutants', type=int, default=200)
parser.add_argument('--prima_split', type=str, default='val')

