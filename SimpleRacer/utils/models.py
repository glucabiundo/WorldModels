import torch
import torch.nn as nn
import torch.nn.functional as F

# Latent space dimension
DIM = 16


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride= 2, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.mu, self.logvar = nn.Linear(1152, DIM), nn.Linear(1152, DIM)
        self.fc = nn.Linear(DIM, 1152)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16 , 4, stride = 2, padding =1), 
            nn.ReLU6(),
            nn.ConvTranspose2d(16, 3, 4, stride= 2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x.permute(0,3,2,1))
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1,32,6,6)).permute(0,2,3,1)

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = mu + torch.rand_like(mu) * torch.exp(0.5*logvar)
        return self.decode(z), mu, logvar

