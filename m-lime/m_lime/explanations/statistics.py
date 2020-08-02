
from collections import defaultdict
import numpy as np
np.seterr('raise')

from sklearn.linear_model import SGDRegressor
from sklearn.metrics.pairwise import euclidean_distances

from density_lime.explanations.base import ExplainerBase
from density_lime.explanations.base import ConFavExaples
from density_lime.explanations.models.statistics import BasicStatistics


standard_local_models = {
    'BasicStatistics': BasicStatistics
}

            
# TODO: This class can be inherited from a explain object.
class ExplainStatistics(ExplainerBase):
    # TODO: The explanation can be generated in a progressive way,
    #  we could generate more and more instances to minimize some criteria,
    #  one example could be the error of the linear model or a search to find
    #  a prediction from the sample that are desired.

    def __init__(self, model_predict, density, local_model='BasicStatistics', random_state=None, verbose=False):
        """
        Simple class to perform explanation using basic statistics of the predictions aroud an instance.
        This is useful when the space of the features has no ... For instance, text.
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param random_state: seed for random condition.
        :param linear_model: linear model that will be used to generate the explanation.
        See standard_linear_models variable.
        """
        super().__init__(model_predict, density, local_model, random_state, verbose)
        self.local_model_name = local_model
        self.local_algorithm = standard_local_models[self.local_model_name] 



if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    import os.path
    import numpy as np
    import os
    import sklearn.model_selection

    import sklearn

    import sklearn.model_selection
    import sklearn.linear_model
    import sklearn.ensemble
    from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split

    from matplotlib import pyplot as plt
    from pprint import pprint
    
    path_ = '/media/tiagobotari/tiagobotari/data/text/rt-polaritydata/rt-polaritydata'
    def load_polarity(path=path_):
        data = []
        labels = []
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(path, f), 'rb'):
                data.append(line.decode('utf8', errors='ignore').strip())
                labels.append(l)
        return data, labels

    x, y = load_polarity()
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x, y, test_size=.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all, y_train_all, test_size=.1, random_state=42)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)


    class VectorizeText():
        def __init__(self):
            self.count_vect = CountVectorizer()
            self.tf_transformer = TfidfTransformer(use_idf=False)
        def fit(self, x):
            x = self.count_vect.fit_transform(x)
            self.tf_transformer.fit(x)
        def transform(self, x):
            x = self.count_vect.transform(x)
            x = self.tf_transformer.transform(x)
            return x
        
    vect_text = VectorizeText()
    vect_text.fit(x_train)    

    x_vec_train = vect_text.transform(x_train)

    # Train Model
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(x_vec_train, y_train)

    def MNB_predict(texts):
        return clf.predict(vect_text.transform(texts))
    def MNB_predict_prob(texts):
        return clf.predict_proba(vect_text.transform(texts))

    preds = MNB_predict(x_val)
    print('Val accuracy', sklearn.metrics.accuracy_score(y_val, preds))

    print('##############################')
    print('### CREATE INTERPRETATION: ###')
    from density_lime.densities.density_word2vec import DensityWord2Vec

    density = DensityWord2Vec(x_train_all)
    
    print('Training Error:', density.manifold.loss_training)
    print('Similarities')
    print()

    pprint(density.manifold.get_similar_words('a', n_sample=10))

    explainer = ExplainStatistics(model_predict=MNB_predict_prob, density=density)
    print('-------')
    print('Test:')
    x_explain = "the movie's thesis -- elegant technology for the masses -- is surprisingly refreshing ."
    print(x_explain)
    print('predicted - true')
    print(MNB_predict([x_explain]), 1)

    explainer.explain_instance(x_explain=[x_explain], class_index=1)
    
    print('Local Statistics:')
    # print(explainer.local_model.coef_)
    pprint(explainer.local_model.results())