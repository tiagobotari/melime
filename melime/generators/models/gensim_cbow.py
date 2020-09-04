from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import gensim


class GensimCBOW:
    def __init__(self, x=None, tokenize_corpus=True):
        super().__init__()
        self.tokenize_corpus = tokenize_corpus
        if x is None:
            self.vocab_update = False
        else:
            self.vocab_update = True
            if self.tokenize_corpus:
                x = Dataset(sentences=x)
        # self.model = Word2Vec(sentences=None, min_count=1, sg=0, hs=0, negative=5, workers=1)
        self.model = Word2Vec(sentences=x, min_count=1, sg=0, hs=0, negative=5, workers=1, compute_loss=True)

    def fit(self, x, y=None, sample_weight=None, epochs=None):
        if self.tokenize_corpus:
            x = Dataset(sentences=x)
        self.model.build_vocab(sentences=x, update=self.vocab_update)
        self.vocab_update = True
        self.model.train(x, total_examples=self.model.corpus_count, epochs=epochs, compute_loss=True)

    @property
    def loss_training(self):
        return self.model.get_latest_training_loss()

    def get_similar_words(self, word, n_sample=100):
        words_distances = self.model.wv.most_similar_cosmul(word, topn=n_sample)
        return words_distances

    def predict_central_word(self, context_words_list, n_sample=10):
        return self.model.predict_output_word(context_words_list, topn=n_sample)

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model.load(file_name)


class Dataset:
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        for sentence in self.sentences:
            tokens = sentence.split()
            yield tokens

    def __getitem__(self, i):
        tokens = self.sentences[i].split()
        return tokens


def tokenize_corpus(corpus):
    return [phrases.split() for phrases in corpus]
