import warnings
import numpy as np
from collections import defaultdict


class ExplainerBase:

    def __init__(self, model_predict, density, linear_model='SGD', random_state=None, verbose=False):
        self.density = density
        self.random_state = random_state
        self.model_predict = model_predict

        self.predictions_index = set()
        self.predictions_stat = {
            'count': defaultdict(int)
            , 'mean_probability': defaultdict(float)
            , 'std_probability': defaultdict(float)
        }

    def explain_instance(self, x_explain, r=None, class_index=0, n_samples=2000, tol=0.0001, n_samples_max=100):
        """
        Generate a Explanation from statistical Measures.
        :param x_explain: instance to be explained
        :param r: radius of the ball of the neighborhood
        :param class_index: class which an explanation will be created
        :param epochs: number of epochs to generate the linear model
        :param n_samples: number of samples for each epochs
        :param tol: tolerance of the change in the importance
        :param epochs_max: number max of epochs to achieve the tol
        :return: explanation in a dict with importance, see status
        """
        samples_generator, features =  self.density.sample_radius(x_explain, r)
        # The model is created when a explanation is requested, centered on the instance. 
        y_p_explain = self.model_predict(x_explain)[0][class_index] 
        self.local_model = self.local_algorithm(mean=y_p_explain, features=features, tol=tol)
        stats = {}
        con_fav_samples = ConFavExaples()
        for x_set, chi_set in samples_generator:
            y_p = self.model_predict(x_set)[:, class_index]
            con_fav_samples.insert_many(x_set, y_p)
            self.local_model.partial_fit(chi_set, y_p)  
            self.local_model.measure_convergence()
            if self.local_model.convergence:
                break
        if not self.local_model.convergence:
            warnings.warn(
                """Convergence tolerance (tol) was not achieved! 
                The current difference in the importance {:}.""".format(tol))
        return self.results(), con_fav_samples
    
    def results(self):
        result = dict()
        result['stats'] = self.stats()
        result['importance'] = self.local_model.importance
        return result

    def stats_(self, y_p):
        class_index = np.argsort(y_p[:, :], axis=1)
        unique, counts = np.unique(class_index[:, -3:], return_counts=True)
        self.predictions_index.update(unique)
        for key, value in zip(unique, counts):
            self.predictions_stat['count'][key] += value
            self.predictions_stat['mean_probability'][key] += np.mean(y_p[:, key])
            self.predictions_stat['std_probability'][key] += np.std(y_p[:, key])

    def stats(self):
        results = dict()
        for key in self.predictions_index:
            results[key] = {
                'count': self.predictions_stat['count'][key]
                , 'mean_probability': self.predictions_stat['mean_probability'][key]/self.predictions_stat['count'][key]
                , 'std_probability': self.predictions_stat['std_probability'][key]/self.predictions_stat['count'][key]
            }

        return results


# class ImportanceMeasure:

#     def __init__(self, tol):
#         self.tol = tol
#         self.previous = None
#         self.convergence = False

#     def update(self, values):
#         if self.previous is None:
#             self.previous = values
#             self.n_importance = len(values)
#         else:
#             diff = np.sum(np.abs(self.previous - values))/self.n_importance
#             if diff < self.tol:
#                 self.convergence = True
#             self.previous = values


class ConFavExaples(object):

    def __init__(self, n_max=5):
        self.n_max = n_max
        self.y_con = list()
        self.y_fav = list()
        self.samples_con = list()
        self.samples_fav = list()
    
    def insert_many(self, samples, ys):
        for sample, y in zip(samples, ys):
            self.insert(sample, y)

    def insert(self, sample, y):
        if len(self.y_con) < self.n_max:
            self.y_con.append(y)
            self.samples_con.append(sample)        
        else:
            if y > self.y_con[-1]:
                self.y_con[-1] = y
                self.samples_con[-1] = sample
        indices_ = np.argsort(self.y_con).reshape(-1)[::-1]
        self.y_con = [self.y_con[e] for e in indices_]
        self.samples_con = [self.samples_con[e] for e in indices_]

        if len(self.y_fav) < self.n_max:
            self.y_fav.append(y)
            self.samples_fav.append(sample)    
        else:
            if y < self.y_fav[-1]:
                self.y_fav[-1] = y
                self.samples_fav[-1] = sample
        indices_ = np.argsort(self.y_fav).reshape(-1)   
        self.y_fav = [self.y_fav[e] for e in indices_]
        self.samples_fav = [self.samples_fav[e] for e in indices_]
    
    def print_results(self):
        print('Contrary:')
        for e, ee in zip(self.samples_con, self.y_con):
            print(e, ee)
        print('Favarable:')
        for e, ee in zip(self.samples_fav, self.y_fav):
            print(e, ee)
     

