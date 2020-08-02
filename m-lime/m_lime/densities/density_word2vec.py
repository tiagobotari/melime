import sys

sys.path.append('../../')
from enum import Enum
import numpy as np

from density_lime.densities.base_density import Density
from density_lime.densities.models.gensim_cbow import GensimCBOW
from density_lime.densities.models.torch_word2vec import TorchCBOW


models_available = {
    'gensim_CBOW': GensimCBOW
    , 'torch_CBOW': TorchCBOW
}
class ModelsAvailable(Enum):
    GensimCBOW = 'gensim_CBOW'
    TorchCBOW = 'torch_CBOW'

class DensityWord2Vec(Density):
    """
    Density estimation using word2vec.
    """
    def __init__(self, sentences=None, model_type='gensim_CBOW', verbose=True, **kwargs):
        super().__init__()
        self.model_type = model_type
        if self.model_type in models_available:
            self.manifold = models_available[self.model_type](x=sentences)
        self.verbose = verbose

    def fit(self, x, y=None, sample_weight=None, epochs=30): 
        self.manifold.fit(x, y=None, sample_weight=None, epochs=epochs)
        return self

    def predict(self, x, kernel_width=None):
        """
        :param x:
        :param kernel_width:
        :return:
        """
        return None
        
    def sample_radius(self, x_exp, r=None, n_samples=1000, random_state=None):
        """
        Generate random samples from the model.
        """
        tokens_exp = x_exp[0].split()
        samples = dict()
        for pos, word in enumerate(tokens_exp):
            new_sentences = []
            samples[pos] = self.manifold.get_similar_words(word, n_sample=n_samples)
        
        features = [f'{pos}: {word}' for pos, word in enumerate(tokens_exp)]
        samples_data = DatasetSamples(features, tokens_exp, samples)
        
        return samples_data, features
       

    def sample(self, n_samples=1, random_state=None):
        return self.manifold.sample(n_samples=n_samples, random_state=random_state)


class DatasetSamples:

    def __init__(self, features, tokens_exp, samples):
        self.samples = samples
        self.tokens_exp = tokens_exp
        self.features = features
        self.n_words = len(self.features)
        self.n_samples_per_pos = len(samples[0])

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for j in range(self.n_samples_per_pos):
            sentences = []
            features = []
            for pos, feature in enumerate(self.features):
                sentences.append(self.create_sentences(pos, self.samples[pos][j][0]))  
                features.append(feature)
            sentences = np.array(sentences).reshape(-1)
            features = np.array(features).reshape(-1, 1)
            yield sentences, features

    def create_sentences(self, pos, word):
        """
        The samples are created by each postion of the sentence,
        repited times.
        """
        return ' '.join(self.tokens_exp[:pos] + [word] + self.tokens_exp[pos+1:])
    

def tokenize_corpus(corpus):
    return [phrases.split() for phrases in corpus]


