import sys
sys.path.append('..')

from lime.lime_image import ImageExplanation, LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm

import copy
import numpy as np

import sklearn
from skimage.color import gray2rgb
from progressbar import ProgressBar

class DensityImageExplanation(ImageExplanation):
    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """
            Simple extention of `lime.lime_image.ImageExplanation`, which allows for another
            background color than black when `hide_rest` is True.
            Just copied the file from the LIME implementation. See comment, where we did something
            extra.
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
            # FHV adding minimal extra functionality to fix color
            if isinstance(hide_rest, int):
                temp += hide_rest
            # FHV end edit
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class DensityImageExplainer(LimeImageExplainer):
    def __init__(self, density, *args, **kwargs):
        """
        Init function  
        Args:
            density: class with the function sample_around(x, n_samples, batch_size), 
                which produces samples based on the input `x`. One such "density"
                could be the function `data_labels` originally used in the 
                LimeImageExplainer.
        """
        super(DensityImageExplainer, self).__init__(*args, **kwargs)
        self.density = density


    def segment(self, image, segmentation_fn=None, random_seed=None):
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e
        return segments


    def fill(self, image, segments, hide_color=None):
        fill = image.copy()
        if self.density is not None:
            fill = self.density.fill(image, segments)

        elif hide_color is None:
            for x in np.unique(segments):
                fill[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))

        else:
            fill[:] = hide_color

        return fill


    # FH: This code is very similar to the original code. I have only
    #     refactored a couple of functionalities, such that they are
    #     easier to extend later. 
    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, 
                         num_features=100000, 
                         num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: The batch size for predicting new samples
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        # If gray scale (e.g., mnist 28x28), then convert to rgb (28x28x3)
        if len(image.shape) == 2:
            image = gray2rgb(image)

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # FH REFACTOR
        # Refactored functions
        segments    = self.segment(image, segmentation_fn, random_seed)
        fill        = self.fill(image, segments, hide_color=hide_color)
        # END FH REFACTOR - only variable names changed below

        top = labels
        data, labels = self.data_labels(image, fill, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = DensityImageExplanation(image, segments)

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp
