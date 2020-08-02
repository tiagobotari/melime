import warnings
import re
from collections import Counter
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        self.input_dim = self.embedding_dim*self.context_size
        # Create Layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.l1 = nn.Linear(self.context_size*self.embedding_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(-1, self.input_dim)
        y = F.log_softmax(self.l1(embeds), dim=1)
        return y
    
    
class ModelCBOW(object):

    def __init__(
        self
        , vocab_size
        , context_size
        , vocabulary=None
        , word_to_index=None
        , index_to_word=None
        , embedding_dim=12
        , learning_rate=1e-3
        , cuda=True, verbose=True
    ):
        self.verbose = verbose

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.vocabulary = vocabulary
        self.context_size = context_size
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.lr=learning_rate

        self.batch_size = 128
        self.cuda = cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device_cpu = torch.device("cpu")
        
        self.model = CBOW(
            vocab_size=vocab_size
            , embedding_dim=self.embedding_dim
            , context_size=context_size
        ).to(self.device)
        
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.NLLLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.interval_print = 10
    
    def get_vectors(self, words):
        if self.word_to_index is None:
            warnings.warn('word_to_index was not defined.')
            return None                  
        if not isinstance(words, list):
            words = list(words)
        return [self.word_to_index[word] for word in words]
        
    def get_words(self, vectors):
        if self.index_to_word is None:
            warnings.warn('index_to_word was not defined.')
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

#             if self.verbose:
#                 if batch_idx % self.interval_print == 0:
#                     print(f'Train Epoch: {epoch} \tLoss: {current_loss}')
        if self.verbose:
            print(f'Epoch: {epoch} - current_loss loss: {current_loss}')
    
    def predict(self, x):
#         context_idxs = torch.tensor(
#             [self.word_to_ix[x_i] for x_i in x], dtype=torch.long
#         ).to(self.device)
        x = torch.tensor(x).to(self.device)
        print(x)
        print(x.size())
        y = self.model.forward(x)
        print('y:')
        print(y)
        print()
        y_arg = torch.argmax(y)
        y_val, y_ind = y.sort(descending=True)
        indices = [y_ind[i][0] for i in np.arange(0,3)]
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
        print('Loss test set: {:.4f}'.format(test_loss))

    def train_epochs(self, train_loader, epochs):
        for epoch in range(1, epochs + 1):
            self.train(train_loader, epoch)



class ContextDataset(Dataset):
    
    def __init__(self, corpus, win_size=1, transform=None):
        """
        
        """
        self.corpus = corpus
        self.transform = transform
        data = initialize_corpus(self.corpus)
        self.vocabulary_size = data['vocabulary_size']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocabulary = data['vocabulary']
        self.corpus_tokenized = data['corpus_tokenized']
        self.win_size = win_size
        
        self.context, self.target = generate_context_target(
            self.corpus_tokenized, self.word2idx, win_size=win_size
        )
    
    def convert_to_text(self, indices):
        return [self.idx2word[i] for i in indices]
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.context[idx]
        y = self.target[idx]  
        if self.transform:
            x, y = self.transform(x, y)
        return x, y
    
    
def generate_context_target(tokenize_corpus, word2idx, win_size=1):
    context_words = []
    target_words = []
    for text in tokenize_corpus:
        size_text = len(text)
        for i, word in enumerate(text):
            j_initial = i-win_size
            j_final = i+win_size+1
            if j_initial < 0 or j_final > size_text:
                continue
            context = []
            for j in range(j_initial, j_final):
                if j!=i:
                    context.append(word2idx[text[j]])
            context_words.append(context)
            target_words.append(word2idx[word])
    context_words = np.array(context_words).reshape(-1,win_size*2)

    return context_words, target_words
 
    
def initialize_corpus(corpus):
    vocabulary = get_vocabulary(corpus)
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)
    corpus_tokenized = tokenize_corpus(corpus)
    return dict(
        vocabulary=vocabulary
        , vocabulary_size=vocabulary_size
        , word2idx=word2idx
        , idx2word=idx2word
        , corpus_tokenized=corpus_tokenized
    )


def tokenize_corpus(corpus):
    return [phrases.split() for phrases in corpus]
   
    
def get_vocabulary(corpus):
    return set([word for phrase in corpus for word in phrase.split()])