from enum import Enum
import numpy as np

from melime.generators.gen_base import GenBase
from melime.generators.models.gensim_cbow import GensimCBOW
from melime.generators.models.torch_word2vec import TorchCBOW


models_available = {"gensim_CBOW": GensimCBOW, "torch_CBOW": TorchCBOW}


class ModelsAvailable(Enum):
    GensimCBOW = "gensim_CBOW"
    TorchCBOW = "torch_CBOW"


class Word2VecGen(GenBase):
    """
    Density estimation using word2vec.
    """

    def __init__(self, sentences=None, model_type="gensim_CBOW", verbose=True, transformer=True, **kwargs):
        super().__init__()
        self.model_type = model_type
        if self.model_type in models_available:
            self.manifold = models_available[self.model_type](x=sentences)
        self.verbose = verbose
        self.transformer = transformer
        self.generated_data = None

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
        if self.generated_data is None:
            tokens_exp = x_exp[0].split()
            samples = dict()
            for pos, word in enumerate(tokens_exp):
                samples[pos] = self.manifold.get_similar_words(word, n_sample=n_samples * 200)

            self.generated_data = DatasetSamples(tokens_exp, samples)

        return self.generated_data.sample(n_samples)

    def sample(self, n_samples=1, random_state=None):
        return self.manifold.sample(n_samples=n_samples, random_state=random_state)

    def transform(self, x):
        chi = np.array([i for i, _ in enumerate(x[0].split())]).reshape(1, -1)
        return chi


class DatasetSamples:
    def __init__(self, tokens_exp, samples):
        self.samples = samples
        self.tokens_exp = tokens_exp
        self.n_words = len(self.tokens_exp)
        self.n_samples_per_pos = len(samples[0])
        self.n_samples_max = len(samples[0])
        self.j = 0

    def sample(self, n_samples):
        sentences = []
        features = []
        for _ in range(n_samples):
            for pos in range(self.n_words):
                sentences.append(self.create_sentences(pos, self.samples[pos][self.j][0]))
                features.append(pos)
            self.j += 1
            if self.j == self.n_samples_max:
                return None, None
        sentences = np.array(sentences).reshape(-1)
        features = np.array(features).reshape(-1, 1)
        return sentences, features

    def create_sentences(self, pos, word):
        """
        The samples are created by each postion of the sentence,
        repited times.
        """
        return " ".join(self.tokens_exp[:pos] + [word] + self.tokens_exp[pos + 1 :])


def tokenize_corpus(corpus):
    return [phrases.split() for phrases in corpus]
