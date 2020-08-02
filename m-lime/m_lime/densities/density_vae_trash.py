import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional
from torchvision import datasets, transforms

import numpy as np

from density_lime.kernel_density_exp import KernelDensityExp


class DensityVAE(object):

    def __init__(self, kernel='gaussian', bandwidth=0.1, n_components=None):
        self.pca = IncrementalPCA(n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = KernelDensityExp(
            kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=50, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.pca.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample


class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim, nodes_dim=None, n_layers=2):
        # super(Encoder, self).__init__()

        super().__init__()

        self.nodes_dim = nodes_dim
        self.latent_dim = latent_dim

        self.layers = create_layers(input_dim, nodes_dim=nodes_dim, n_layers=n_layers)
        self.activation = functional.relu

        self.mu_layer = nn.Linear(self.nodes_dim, self.latent_dim)
        self.log_var_layer = nn.Linear(self.nodes_dim, self.latent_dim)

    def forward(self, x):
        x_next = x
        for layer in self.layers:
            x_next = self.activation(layer(x_next))

        mu = self.mu_layer(x_next)
        log_var = self.log_var_layer(x_next)

        return mu, log_var

    # def __call__(self, x, *args, **kwargs):
    #     return self.forward(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, nodes_dim=None, n_layers=2):
        super(Decoder, self).__init__()
        self.nodes_dim = nodes_dim
        self.latent_dim = latent_dim
        self.layers = create_layers(latent_dim, nodes_dim=nodes_dim, n_layers=n_layers)
        self.activation = functional.elu
        self.output_layer = nn.Linear(self.nodes_dim, output_dim)

    def forward(self, x):
        x_next = x
        for layer in self.layers:
            x_next = self.activation(layer(x_next))
        out_put = self.output_layer(x_next)

        return torch.sigmoid(out_put)

    # def __call__(self, x, *args, **kwargs):
    #     return self.forward(x)


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, nodes_dim=None, n_layers=2, cuda=True, device=None):
        # super(VAE, self).__init__()
        super().__init__()

        self.epochs = 50
        self.log_interval = 10
        self.device = torch.device("cuda" if cuda else "cpu")
        self.input_dim = input_dim
        # model = self.to(self.device)
        # self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.encoder = Encoder(
            input_dim=input_dim, latent_dim=latent_dim, nodes_dim=nodes_dim, n_layers=n_layers) #.to(self.device)

        self.decoder = Decoder(
            output_dim=input_dim, latent_dim=latent_dim, nodes_dim=nodes_dim, n_layers=2) #.to(device)

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.input_dim))
        z = self.reparameterization(mu, log_var)
        return self.decoder(z), mu, log_var


class ModelVAE(object):

    def __init__(self, input_dim, latent_dim=12, nodes_dim=20, n_layer=2, cuda=True):
        self.epochs = 50
        self.log_interval = 10
        self.device = torch.device("cuda" if cuda else "cpu")
        self.input_dim = input_dim

        self.model = VAE(input_dim, latent_dim=latent_dim, nodes_dim=nodes_dim, device=self.device, cuda=cuda).to(self.device)
        # self.optimizer = optim.Adam(
        #     [self.model.parameters(), self.model.encoder.parameters(), self.model.decoder.parameters()], lr=1e-3)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD

    def train_(self, epoch, train_loader):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))


def create_layers(input_dim, nodes_dim, n_layers):
        layers = list()
        input_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_features=input_dim, out_features=nodes_dim))
            input_dim = nodes_dim

        return layers


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                         recon_batch.view(batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                            'results_reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':
    from torchvision import datasets, transforms

    epochs = 20
    cuda = False
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = ModelVAE(input_dim=784, latent_dim=12, nodes_dim=400, n_layer=2, cuda=cuda)
    for epoch in range(1, epochs + 1):
        model.train_(epoch, train_loader)
