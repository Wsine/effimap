from data.noisycifar.data.cifar import CIFAR10 as ORICIFAR10
from data.noisycifar.data.cifar import CIFAR100 as ORICIFAR100


class NCIFAR10(ORICIFAR10):
    def __getitem__(self, index):
        img, target, _ =  super().__getitem__(index)
        if self.train:
            clean_target = self.train_labels[index]
            target = (target, clean_target)
        return img, target


class NCIFAR100(ORICIFAR100):
    def __getitem__(self, index):
        img, target, _ =  super().__getitem__(index)
        if self.train:
            clean_target = self.train_labels[index]
            target = (target, clean_target)
        return img, target


if __name__ == '__main__':
    dataset = NCIFAR100(
        root='data', train=True, download=False,
        noise_type='noisy_label',
        noise_path='data/noisycifar/data/CIFAR-100_human.pt')
    print(dataset[0])
