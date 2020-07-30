import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional


from density_lime.densities.base_density import Density


class DensityCNNVAE(Density):

    def __init__(self, image_channels, latent_dim, cuda=True, verbose=False, **kwargs):
        self.model = ModelVAE(image_channels, latent_dim=latent_dim, cuda=cuda, verbose=verbose, **kwargs)
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
            mu_p, log_var_p = self.model.model.encode(x_exp_tensor)
            ones = torch.ones(n_samples).to(self.model.device)
            mu_m = torch.ger(ones, mu_p.view(-1))
            # TODO: TB: I am not sure if is better or not multiply the distance r by std_r.
            # TODO: TB: preliminary tests indicate that is better to not use std_r.
            # std_r = torch.exp(0.5 * log_var_p).to(self.model.device)
            noise = torch.rand(n_samples, self.model.latent_dim).to(self.model.device) * r  # std_r *
            mu_m = mu_m + noise
            z = self.model.model.reparameterize(mu_m, log_var_p)
            x_sample = self.model.model.decode(z)

        # Clean cache torch.
        # TODO: TB: what is the best practice to clean cash, is it really necessary?
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

    def __init__(self, image_channels, latent_dim=256, cuda=True, verbose=False, **kwargs):
        self.verbose = verbose
        self.latent_dim = latent_dim
        self.image_channels = image_channels

        self.batch_size = 128
        self.cuda = cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device_cpu = torch.device("cpu")

        self.model = VAE(image_channels=self.image_channels, latent_dim=latent_dim)
        if self.cuda:
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # Variable to print on the screen.
        self.log_interval = 1000

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)  # .reshape(-1, self.input_dim)
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            # loss = self.model.loss_function(recon_batch, data, mu, log_var)
            loss = self.model.loss_function_2(recon_batch, data, mu, log_var)

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
            self.scheduler.step()


class VAE(nn.Module):

    def __init__(self, image_channels=3, kernel_size=3, stride=1, latent_dim=32):
        super().__init__()
        self.verbose = True
        self.image_channels = image_channels

        self.latent_dim = latent_dim
        self.channels = [32, 64, 128, 256, 512]

        # Encoder
        self.layers_encoder = nn.ModuleList(self.create_layers_encoder(image_channels=image_channels))

        # Mu and log_var
        in_linear_layer = self.channels[-1]  # *4
        self.fc_mu = nn.Linear(in_linear_layer, self.latent_dim)
        self.fc_log_var = nn.Linear(in_linear_layer, self.latent_dim)

        self.fc_out = nn.Linear(self.latent_dim, in_linear_layer)

        # Decoder
        self.layers_decoder = nn.ModuleList(
            self.create_layers_decoder(
                image_channels=image_channels
            )
        )

    def encode(self, x):
        x_in = x
        for layer in self.layers_encoder:
            # TODO: doubt!! no functional here, not sure what is the best option
            x_in = layer(x_in)
        x_in = torch.flatten(x_in, start_dim=1)
        return self.fc_mu(x_in), self.fc_log_var(x_in)

    def decode(self, z):
        z = self.fc_out(z)
        z = z.view(-1, self.channels[-1], 1, 1)
        x_in = z
        for layer in self.layers_decoder:
            # TODO: doubt!! no functional here, not sure what is the best option
            x_in = layer(x_in)
        return x_in

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def create_layers_encoder(self, image_channels):
        # TODO: Would be nice to have some options here.
        # TODO: doubt!!! I choose to use the Elu layer here. not sure about this.
        # TODO: I am thinking to put a batch normalization between the layers.
        out_chanels = self.channels
        layers = list()
        for out_chanel in out_chanels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(image_channels, out_chanel, kernel_size=3, stride=2, padding=1)
                    , nn.BatchNorm2d(out_chanel)
                    , nn.LeakyReLU()
                )
            )
            image_channels = out_chanel
        return layers

    def create_layers_decoder(self, image_channels):
        # TODO: Would be nice to have some options here.
        # TODO: doubt!!! I choose to use the Elu layer here. not sure about this.
        # TODO: I am thinking to put a batch normalization between the layers.
        out_channels = self.channels[:]

        out_channels.reverse()

        layers = list()
        for in_chanel, out_chanel in zip(out_channels[:-1], out_channels[1:]):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_chanel, out_chanel, kernel_size=3, stride=2, padding=1, output_padding=1
                        # , bias=False  # TODO: I want to include this!
                    )
                    , nn.BatchNorm2d(out_chanel)
                    , nn.LeakyReLU()
                )
            )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[-1],
                                   out_channels[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                # TODO: I remove that, but not sure what to do, I want to improve the resolution.
                #                             nn.BatchNorm2d(out_chanels[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels[-1], out_channels=self.image_channels,
                          kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        return layers

    @staticmethod
    def loss_function(recons, input_, mu, log_var, kld_weight=1):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = functional.mse_loss(recons, input_)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return loss  # {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def loss_function_2(self, recon_x, x, mu, log_var):
        # TODO: check if this is the best loss
        bce = functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kld


if __name__ == '__main__':
    import torch
    import torch.utils.data
    from torchvision import datasets, transforms
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision

    epochs = 10
    cuda = torch.cuda.is_available()
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    device = torch.device("cuda" if cuda else "cpu")
    device_cpu = torch.device("cpu")

    transform_data = transforms.Compose([transforms.Resize(32)
                                         , transforms.ToTensor()])

    folder_data = '/media/tiago/tiagobotari/data'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(folder_data, train=True, download=False,
                       transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            folder_data, train=False, download=False
            , transform=transform_data), batch_size=batch_size, shuffle=True, **kwargs
    )

    density = DensityCNNVAE(image_channels=1, latent_dim=512)
    density.fit(train_loader, epochs=10)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    x_explain = example_data[0][0].numpy()
    print(x_explain.shape())
    x_sample = density.sample_radius(x_exp=x_explain.reshape(-1, 784), r=0.1, n_samples=15000, random_state=None)


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updateda


    inputs = x_sample
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out)

