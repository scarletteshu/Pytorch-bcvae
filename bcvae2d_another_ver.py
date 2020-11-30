import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ConditionalVAE(nn.Module):
   
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 gamma: float = 100.,
                 beta: int = 7,
                 cur_device = torch.device('cpu:0'),
                 batch_size : int = 144,
                 **kwargs) -> None:

        super(ConditionalVAE, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.gamma = gamma
        self.beta = beta
        self.cur_device = cur_device

        modules = []

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),

            nn.Linear(256, latent_dim * 2),  # B, z_dim*2
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # B, nc, 64, 64
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        # Returns a tensor with the same size as std that is filled with random numbers from a normal distribution
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, input: Tensor):
        return self.encoder(input)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def forward(self, input: Tensor, **kwargs):
        distribution = self.encode(input)
        mu = distribution[:, :self.latent_dim]
        log_var = distribution[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        return recons, mu, log_var

    def lossFunc(self, recons, target, mu, log_var, num_iter):
        """
        c [0, 25], linearly increasing
        as iteration continue, loss will be up to gamma*0.0008680*|kldloss-25| + recons_loss
        """

        recons_loss = F.binary_cross_entropy(recons, target, reduction='sum').div(self.batch_size)
        #recons_loss = F.mse_loss(recons, target, reduction='sum').div(self.batch_szie)
        kld  = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        self.C_max = self.C_max.to(self.cur_device)
        C    = torch.clamp(self.C_max / self.C_stop_iter * num_iter, 0, self.C_max.data[0])
        loss = recons_loss + self.gamma * (kld - C).abs()  # * kld_weight

        return loss, recons_loss, kld

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
