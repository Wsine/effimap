import torch
from torch import nn
from torchattacks.attack import Attack

from arguments import parser
from dataset import load_dataloader
from model import load_model
from utils import check_file_exists, get_device, get_output_path, guard_folder


class PGDAttack(Attack):
    def __init__(self, model, criterion,
                 eps=0.3, alpha=2/255, steps=40, random_start=True):
        super().__init__("PGDAttack", model)
        self.criterion = criterion
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def generate_with_PGD_attack(ctx, model, dataloader):
    if ctx.task == 'clf':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplemented

    atk = PGDAttack(model, criterion, eps=8/255)
    save_path = get_output_path(ctx, 'pgd_adversarial_samples.pt')
    atk.save(dataloader, save_path=save_path)


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)

    device = get_device(ctx)
    trainloader = load_dataloader(ctx, split='train')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    output_name = 'pgd_adversarial_samples.pt'
    if not check_file_exists(ctx, output_name):
        generate_with_PGD_attack(ctx, model, trainloader)


if __name__ == '__main__':
    main()
