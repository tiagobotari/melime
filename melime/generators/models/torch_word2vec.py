import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import re
from collections import Counter

torch.manual_seed(1)


class TorchCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size=1, hidden=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden = hidden

        self.input_dim = self.embedding_dim * self.context_size
        # Create Layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        if self.hidden:
            self.hidden_layer_size = 512
            self.hidden_layer = nn.Linear(self.context_size * self.embedding_dim, self.hidden_layer_size)
            self.l1 = nn.Linear(self.hidden_layer_size, vocab_size)
        else:
            self.l1 = nn.Linear(self.context_size * self.embedding_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(-1, self.input_dim)
        if self.hidden:
            embeds = F.elu(self.hidden_layer(embeds))
        y = F.log_softmax(self.l1(embeds), dim=1)
        return y


class ModelCBOW(object):
    def __init__(
        self,
        vocab_size,
        context_size,
        vocabulary=None,
        word_to_index=None,
        index_to_word=None,
        embedding_dim=12,
        learning_rate=1e-3,
        cuda=True,
        optimizer=None,
        verbose=True,
    ):
        self.verbose = verbose

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.vocabulary = vocabulary
        self.context_size = context_size
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.lr = learning_rate

        self.batch_size = 128
        self.cuda = cuda
        self.kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device_cpu = torch.device("cpu")

        self.model = CBOW(vocab_size=vocab_size, embedding_dim=self.embedding_dim, context_size=context_size).to(
            self.device
        )

        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.NLLLoss()
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer
        #         self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.interval_print = 10

    def get_vectors(self, words):
        if self.word_to_index is None:
            warnings.warn("word_to_index was not defined.")
            return None
        if not isinstance(words, list):
            words = list(words)
        return [self.word_to_index[word] for word in words]

    def get_words(self, vectors):
        if self.index_to_word is None:
            warnings.warn("index_to_word was not defined.")
            return None
        if not isinstance(vectors, list):
            vectors = list(vectors)
        return [self.index_to_word[vector] for vector in vectors]

    def get_embeding(self, word):
        index_w = self.get_vectors(word)
        index_w = torch.tensor(index_w).view(-1).to(self.device)
        x_ = self.model.embedding(index_w)
        return x_

    def train(self, train_loader, epoch, tol=0.1):
        # TODO: implement a early stop.
        self.model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_p = self.model(x).view(x.size()[0], -1)
            loss = self.loss_function(y_p, y)
            loss.backward()
            current_loss = loss.item()
            train_loss += current_loss
            self.optimizer.step()

        if self.verbose:
            print(f"Epoch: {epoch} - current_loss loss: {current_loss}")

    def predict(self, x):
        x = torch.tensor(x).to(self.device)
        print(x)
        print(x.size())
        y = self.model.forward(x)
        print("y:")
        print(y)
        print()
        y_arg = torch.argmax(y)
        y_val, y_ind = y.sort(descending=True)
        indices = [y_ind[i][0] for i in np.arange(0, 3)]
        return indices

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, log_var).item()

        test_loss /= len(test_loader.dataset)
        print("Loss test set: {:.4f}".format(test_loss))

    def train_epochs(self, train_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.train(train_loader, epoch)

#     def loss_function(self, y_pred, y_true):
#         # Negative-log-likelihood - logsoftmax
#         # loss = F.nll_loss(y_pred, y_true)
#         return loss
