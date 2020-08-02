import numpy as np


class BasicStatistics:

    def __init__(self, mean, features, tol=0.01):
        super().__init__()
        self.tol = tol
        self.previous_convergence = None
        self.n_previous_convergence = None
        self.convergence = False
        self.mean = mean
        self.features = features
        self.values = {e: [] for e in self.features}

    def measure_convergence(self):
        values = self._coef_mean
        if self.previous_convergence is None:
            self.previous_convergence = values
            self.n_previous_convergence = len(values)
        else:
            print('self.previous_convergence')
            print(self.previous_convergence)
            print(values)
            diff = (np.sum(np.abs(self.previous_convergence - values))
                    /self.n_previous_convergence
                   )
            if diff < self.tol:
                self.convergence = True
            self.previous_convergence = values

    @property
    def _coef_mean(self):
        results = self.calculate()
        return np.array([*results['mean'].values()])

    @property
    def importance(self):
        return self.calculate()
        
    def results(self):
        results = self.calculate() 
        return results

    def partial_fit(self, x_set, y_set):
        # TODO: I need to improve this.
        for x_i, y_i in zip(x_set, y_set):
            self.values[x_i[0]].append(y_i)
        
    def calculate(self):
        mean = {}
        median = {}
        std = {}
        for key, values in self.values.items():
            if len(values) == 0:
                mean[key] = 0
                median[key] = 0
                std[key] = 0
                continue
            values_c = values-self.mean
            mean[key] = np.mean(values_c)
            median[key] = np.median(values_c)
            std[key] = np.std(values_c)

        return dict(
            mean=self.normalize_max(mean)
            , median=self.normalize_max(median)
            , std=self.normalize_max(std)
            )

    @staticmethod
    def normalize_max(dict_in):
        max_values = np.max(np.abs([*dict_in.values()]))
        if max_values == 0.0:
            return dict_in
        dict_out = {key: value/max_values for key, value in dict_in.items()}
        return dict_out