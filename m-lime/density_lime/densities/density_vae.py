import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional

from density_lime.densities.base_density import Density


class DensityVAE(Density):

    def __init__(self, input_dim=None, verbose=False, **kwargs):
        self.model = ModelVAE(input_dim=input_dim, verbose=verbose, **kwargs)
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

        self.batch_size = 128
        self.cuda = cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device_cpu = torch.device("cpu")

        self.model = VAE(self.input_dim, nodes_dim=nodes_dim, n_layers=n_layers, latent_dim=latent_dim, device=self.device).to(self.device)
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
            print('====> Epoch: {} Average loss: {:.4f}'.format(
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


CUDA = True
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 10
no_of_sample = 10
from torch.autograd import Variable
from torch.nn import functional as F
# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 20


class VAECNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=ZDIMS)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=ZDIMS)
        self.relu = nn.ReLU()

        # For decoder

        # For mu
        self.fc1 = nn.Linear(in_features=20, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

    def encode(self, x: Variable) -> (Variable, Variable):

        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu_z = F.elu(self.fc11(x))
        mu_z = self.fc12(mu_z)

        logvar_z = F.elu(self.fc21(x))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> Variable:

        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = F.sigmoid(self.conv_t2(x))

        return x.view(-1, 784)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar
        # return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?
        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 784))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= BATCH_SIZE * 784

        return BCE + KLD


class VAE(nn.Module):
    def __init__(self, input_dim, nodes_dim=400, n_layers=2, latent_dim=12, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nodes_dim = nodes_dim
        self.device = device
        # Encoder
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
        std = torch.exp(0.5 * log_var)
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