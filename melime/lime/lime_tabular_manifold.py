"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import copy
import sklearn.preprocessing
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

from lime.lime_tabular import *
from melime.generators.kde_gen import KDEGen
from melime.generators.kdepca_gen import KDEPCAGen, KDEKPCAGen


class LimeTabularExplainerManifold(LimeTabularExplainer):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None
                 , manifold='kde'
                 , manifold_params={}):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
            manifold: manifold is a object that supply the neighborhood data for explanation;
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats

        # TODO: I think I included this, need to check, why I created this variable, I think it was created to plot the
        # TODO: plot the data
        self.data = None

        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(training_data, self.categorical_features,
                                               self.feature_names, labels=training_labels,
                                               data_stats=self.training_data_stats)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if(self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)

        # TODO: TB: It may starts here. Ok, here we have a problem,
        # TODO: This routine do not accept sample, only accept a kernel.
        # TODO: Two possible options: (1) change lime, or
        # TODO: (2) create a kernel function from KernelDensityExp.

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)

        # Copy scaler for kde.
        self.scaler_original = copy.deepcopy(self.scaler)

        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

        # KDE and others.
        # Number_of_samples number of instances to fit the KDE.
        number_of_samples = 50000
        self.data_sampled = None
        x = self.scaler_original.transform(training_data)

        if x.shape[0] > number_of_samples:
            ind = np.random.randint(0, x.shape[0], size=number_of_samples)
            x = x[ind]

        # TODO: automatically estimate bandwidth
        self.bandwidth = 0.1

        if manifold == 'kde':
            self.manifold = KDEGen(
            kernel='gaussian', bandwidth=self.bandwidth, **manifold_params).fit(x)
        elif manifold == 'kde-pca':
            # TODO: Chose of n_components is arbitrary.
            n_components = int(np.sqrt(x.shape[1]))
            self.manifold = KDEPCAGen(
                kernel='gaussian', bandwidth=self.bandwidth, n_components=n_components, **manifold_params).fit(x)
        elif manifold == 'kde-kpca':
            # TODO: Chose of n_components is arbitrary.
            n_components = int(np.sqrt(x.shape[1]))
            self.manifold = KDEKPCAGen(
                kernel='gaussian', bandwidth=self.bandwidth, n_components=n_components, **manifold_params).fit(x)
        else:
            raise Exception('The manifold value should be a valid options: kde, kde-pca, and kde-kpca.')

    def explain_instance_manifold(self,
                                  data_row,
                                  predict_fn,
                                  labels=(1,),
                                  top_labels=None,
                                  num_features=10,
                                  num_samples=5000,
                                  distance_metric='euclidean',
                                  model_regressor=None
                                  , r_density=None
                                  , n_min_kernels=None
                                  , scale_data=True
                                  ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            # TODO: TB: check this variables and see with the scale_data=False is a good choice.
            r_density: radius to find the neighborhood
            n_min_kernels: number of kernels, maybe move from here
            scale_data: boolean value to produce the interpretation in the original feature space

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        if isinstance(n_min_kernels, float) and n_min_kernels <=1.0:
            n_min_kernels = int(x.shape[0] * 0.1)
            data, inverse = self.__data_inverse_manifold(data_row, num_samples, n_min_kernels=n_min_kernels)
        elif isinstance(n_min_kernels, int):
            data, inverse = self.__data_inverse_manifold(data_row, num_samples, n_min_kernels=n_min_kernels)
        elif r_density is not None:
            data, inverse = self.__data_inverse_manifold(data_row, num_samples, r_density=r_density)
        else:
            raise Exception('The r_density or n_min_kernels should be defined.')

        self.data = inverse
        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            # TODO: TB: I believe that scale the data is not good practice
            if scale_data:
                scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
            else:
                scaled_data = data

        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse_manifold(self, data_row, num_samples, n_min_kernels=None, r_density=None):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        x_explain_scaled = self.scaler_original.transform(data_row.reshape(1, -1))
        # TODO: implement for sparse.
        if is_sparse:
            ValueError("Not Implemented to sparse, you should convert that.")

        if self.sample_around_instance:
            # TODO: There is a problem when the selected ball radius has no points, maybe we should include some
            # TODO: initial density and sum the kernel.

            if r_density is not None:
                data = self.scaler_original.inverse_transform(self.manifold.sample_radius(
                    x_exp=x_explain_scaled.reshape(1, -1), r=r_density, n_samples=num_samples, random_state=None))
            else:
                data = self.scaler_original.inverse_transform(
                    self.manifold.sample_radius(
                        x_exp=x_explain_scaled.reshape(1, -1)
                        , n_min_kernels=n_min_kernels
                        , n_samples=num_samples
                        , random_state=None
                    ))
        else:
            data = self.scaler_original.inverse_transform(
                self.manifold.sample(n_samples=num_samples, random_state=None, scaler=self.scaler))
        # TODO: implement for categorical data in some way.
        data[0] = data_row.copy()
        inverse = data.copy()

        return data, inverse

    def plot_samples(self, ax, model=None, **kwargs):
        if self.data is not None:
            if model is None:
                return self.plot(self.data, ax, s=1, **kwargs)
            else:
                return self.plot(self.data, ax, s=1, y=model.predict(self.data), **kwargs)

    def plot(self, x, ax=None, alpha=1.0, figsize=(10, 10), y=None, **kwargs):
        import itertools
        n_cols = len(self.feature_names)
        indices_cols = range(n_cols)

        selections = list(itertools.combinations_with_replacement(indices_cols, 2))
        # n_cols = n_#int(len(selections)/2)

        if ax is None:
            fig, ax = plt.subplots(n_cols, n_cols, sharex="col", sharey="row", squeeze=False, figsize=figsize)
        for i, sel in enumerate(selections):
            col1 = sel[0]
            col2 = sel[1]
            axi = ax[col1, col2]
            if y is not None:
                cp = axi.scatter(x[:, col2], x[:, col1], alpha=alpha, c=y,  **kwargs)
            else:
                cp = None
                axi.scatter(x[:, col2], x[:, col1], alpha=alpha, c=y,  **kwargs)
            # axi.text(
            #     0, 1, '{} - {}'.format(col1, col2), fontsize=12, fontweight="bold", va="bottom", ha="left"
            #     , transform=axi.transAxes)
            axi = ax[col2, col1]

            axi.scatter(x[:, col1], x[:, col2], alpha=alpha, c=y, **kwargs)

        for i, label in enumerate(self.feature_names):
            ax[-1, i].set_xlabel(label)
            ax[i, 0].set_ylabel(label)
        # plt.tight_layout()
        if cp is not None:
            return ax, cp
        return ax
