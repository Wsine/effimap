from typing import List, TypeVar
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


Tensor = TypeVar('torch.tensor')  # type: ignore


class VanillaVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_img_size: int,
                 latent_dim: int,
                 hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            in_img_size = in_img_size // 2

        self.hidden_size = in_img_size
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.hidden_size * self.hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.hidden_size * self.hidden_size, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.hidden_size * self.hidden_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.hidden_size, self.hidden_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor, expand: int=None ) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)  # type: ignore
        if not expand:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn(expand, std.size(1))
        return eps * std + mu  # type: ignore

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: str) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples  # type: ignore

    def generate(self, x: Tensor, num_samples: int) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [1 x C x H x W]
        :return: (Tensor) [num_samples x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, expand=num_samples)
        samples = self.decode(z)
        return samples  # type: ignore


class TestVanillaVAE(unittest.TestCase):
    def setUp(self) -> None:
        self.model = VanillaVAE(3, 32, 10)

    def test_summary(self):
        print('==== test_summary ====')
        print(summary(self.model, (3, 32, 32), device='cpu'))

    def test_forward(self):
        print('==== test_forward ====')
        x = torch.randn(16, 3, 32, 32)
        y = self.model(x)
        print("Model Output size:", y[0].size())

    def test_loss(self):
        print('==== test_loss ====')
        x = torch.randn(16, 3, 32, 32)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N=0.005)
        print(loss)

    @torch.no_grad()
    def test_generate(self):
        print('==== test_generate ====')
        self.model.eval()
        x = torch.randn(1, 3, 32, 32)
        samples = self.model.generate(x, num_samples=200)
        print(samples.size())


if __name__ == '__main__':
    unittest.main()

