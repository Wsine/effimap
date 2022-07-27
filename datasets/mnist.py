import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    mean, std = (0.1307,), (0.3081,)
    trsf = [T.ToTensor(), T.Normalize(mean, std)]
    dataset = torchvision.datasets.MNIST(
        root=opt.data_dir, train=train,
        transform=T.Compose(trsf), **kwargs
    )
    return dataset


