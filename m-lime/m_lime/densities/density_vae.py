import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional

from density_lime.densities.base_density import Density


class DensityVAE(Density):

    def __init__(self, input_dim=None, verbose=False, model=None, **kwargs):
        if model is None:
            self.model = ModelVAE(input_dim=input_dim, verbose=verbose, **kwargs)
        else:
            self.model = model
        self.manifold = None

    def fit(self, x, epochs=10):
        self.model.train_epochs(train_loader=x, epochs=epochs)
        return self

    def load_manifold(self, path):
        self.model.model = torch.load(path)
        self.model.model.eval()
        return self

    def save_manifold(self, path):
        torch.save(self.model.model, path)

    def sample_radius(self, x_exp, r=None, n_samples=1000, random_state=None):
        with torch.no_grad():
            x_exp_tensor = torch.from_numpy(x_exp).to(self.model.device)
            mu_p, log_var_p = self.model.model.encode(x_exp_tensor.reshape(-1))
            ones = torch.ones(n_samples).to(self.model.device)
            mu_m = torch.ger(ones, mu_p)
            # TODO: TB: I am not sure if is better or not multiply the distance r by std_r.
            # TODO: TB: preliminary tests indicate that is better to not use std_r.
            # std_r = torch.exp(0.5 * log_var_p).to(self.model.device)
            noise = torch.rand(n_samples, self.model.latent_dim).to(self.model.device)*r  # std_r *
            mu_m = mu_m + noise
            z = self.model.model.reparameterize(mu_m, log_var_p)
            x_p = self.model.model.decode(z)
            x_sample = x_p.reshape(-1, self.model.input_dim).to(self.model.device_cpu).detach().numpy()
            # Clean cache torch.
            # TODO: TB: what is the best practice?
        del x_p
        del noise
        del mu_m
        torch.cuda.empty_cache()

        return x_sample

    def sample(self, n_samples=1, random_state=None):
        # TODO: Need to be implemented.
        pass
        # x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        # x_sample = self.pca.inverse_transform(x_sample_pca)
        # return x_sample


class ModelVAE(object):

    def __init__(self, input_dim, nodes_dim=400, n_layers=2, latent_dim=12, cuda=True, verbose=False):
        self.verbose = verbose
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.nodes_dim = nodes_dim
        
        self.batch_size = 128
        self.cuda = cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device_cpu = torch.device("cpu")

        self.model = VAE(
            self.input_dim
            , nodes_dim=self.nodes_dim
            , n_layers=self.n_layers
            , latent_dim=self.latent_dim
            , device=self.device
            ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.log_interval = 10

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device).reshape(-1, self.input_dim)
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if self.verbose:
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item() / len(data)))

        if self.verbose:
            print('Epoch: {} - Mean loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, log_var).item()

        test_loss /= len(test_loader.dataset)
        print('Loss test set: {:.4f}'.format(test_loss))

    def train_epochs(self, train_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.train(train_loader, epoch)

    def loss_function(self, recon_x, x, mu, log_var):
        # TODO: check if this is the best loss
        bce = functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kld


class VAE(nn.Module):
    def __init__(self, input_dim, nodes_dim=400, n_layers=2, latent_dim=12, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nodes_dim = nodes_dim
        self.device = device
        
        self.layers_encoder = nn.ModuleList(self.create_layers(self.input_dim, self.nodes_dim, n_layers))

        # Mu and log_var
        self.fc_mu = nn.Linear(self.nodes_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.nodes_dim, self.latent_dim)

        # Decoder
        self.layers_decoder = nn.ModuleList(self.create_layers(self.latent_dim, self.nodes_dim, n_layers))

        self.output = nn.Linear(self.nodes_dim, self.input_dim)

    def encode(self, x):
        x_in = x
        for layer in self.layers_encoder:
            x_in = functional.elu(layer(x_in))
        return self.fc_mu(x_in), self.fc_log_var(x_in)

    def decode(self, z):
        x_in = z
        for layer in self.layers_decoder:
            x_in = functional.elu(layer(x_in))
        return torch.sigmoid(self.output(x_in))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    @staticmethod
    def create_layers(input_dim, nodes_dim, n_layers):
        layers = list()
        for i in range(n_layers):
            layers.append(nn.Linear(in_features=input_dim, out_features=nodes_dim))
            input_dim = nodes_dim

        return layers


if __name__ == '__main__':
    from torchvision import datasets, transforms
    import torchvision
    import torchvision.transforms as transforms

    CUDA = True
    SEED = 1
    BATCH_SIZE = 128
    LOG_INTERVAL = 10
    EPOCHS = 10
    no_of_sample = 10

    # connections through the autoencoder bottleneck
    # in the pytorch VAE example, this is 20
    ZDIMS = 20


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    folder_data = '/home/tiago/projects/density-lime/src/playground/kde/data'
    trainset = torchvision.datasets.CIFAR10(root=folder_data, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=folder_data, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    density = DensityVAE(input_dim=3072, verbose=True)
    density.fit(trainloader)


    exit()
    from torchvision import datasets, transforms


    epochs = 5
    cuda = True
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    density = DensityVAE(input_dim=784, verbose=True)
    density.fit(train_loader, epochs=4)

    for data in test_loader:
        x_1 = (data)
        break

    x_img = x_1[0][1].reshape(28, 28)
    x_img_plot = x_img.cpu().numpy()
    # model = ModelVAE(input_dim=784, n_layers=1)
    # model.train_epochs(epochs=10)
    density.sample_radius(x_exp=x_img_plot.reshape(-1, 784), r=10, n_samples=1, random_state=None)

    import matplotlib.pyplot as plt

    # img = sample.view(64, 1, 28, 28)
    fig, ax1 = plt.subplots(1, 1)
    # ax1.imshow(img[50][0], interpolation = 'none')
    ax1.imshow(x_img_plot, interpolation='none')
    # ax1.set_title('Digit: {}'.format(y))
    plt.show()