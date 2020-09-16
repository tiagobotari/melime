import os
if not os.path.exists('submodules/innvestigate/innvestigate'):
    raise IOError("You have not cloned `innvestigate` submodule.\nRun `git submodule update --init` in the terminal")

import sys
sys.path.append('submodules/innvestigate')
sys.path.append('.')

import imp
import numpy as np
import os

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.utils as iutils
from innvestigate.analyzer.base import AnalyzerBase
import innvestigate.utils.visualizations as ivis

import matplotlib.pyplot as plt

from melime.generators.vae_gen import VAEGen
from melime.explainers.explainer import Explainer

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "submodules/innvestigate/examples/utils.py")
mnistutils = imp.load_source("utils_mnist", "submodules/innvestigate/examples/utils_mnist.py")
def graymap(X):
    return ivis.graymap(np.abs(X))


# # # # # #
# DATA 
# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
data_not_preprocessed = mnistutils.fetch_data()

# Create preprocessing functions
input_range = [0, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

# Preprocess data
data = ( 
    # X_train, y_train
    preprocess(data_not_preprocessed[0]), data_not_preprocessed[1],
    # X_test, y_test
    preprocess(data_not_preprocessed[2]), data_not_preprocessed[3]
)

num_classes = len(np.unique(data[1]))
label_to_class_name = [str(i) for i in range(num_classes)]


# # # # # #
# Model 
# Create & train model
if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)

path_ = "experiments/"
os.makedirs(f"{path_}pretrained", exist_ok=True)
model_path = f"{path_}pretrained/cnn.keras"

# Hack to make `model_wo_softmax` work for models loaded from disc
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])

if os.path.exists(model_path): 
    print("# " * 25)
    print("# Loading pretrained model")
    print("# " * 25)
    model = keras.models.load_model(model_path)
else: 
    print("# " * 25)
    print("# Training new CNN Model to roughly 99% accuracy")
    print("# " * 25)
    scores = mnistutils.train_model(model, data, batch_size=128, epochs=20)
    print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
    model.save(model_path)

# # # # # #
# Analysis setup 
# Scale to [0, 1] range for plotting.
def input_postprocessing(X):
    return revert_preprocessing(X) / 255

noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input

# Configure analysis methods and properties
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE                            Store fit
    # Show input
    ("input",                 {},                       input_postprocessing,      "Input",                         False),

    # Function
    ("gradient",              {"postprocess": "abs"},   graymap,                    "Gradient",                      False),
    ("smoothgrad",            {"noise_scale": noise_scale,
                               "postprocess": "square"},graymap,                    "SmoothGrad",                    False),

    # Signal
    ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet",                     False),
    ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",               False),
    ("pattern.net",           {"pattern_type": "relu"}, mnistutils.bk_proj,        "PatternNet",                    True),

    # Interaction
    ("pattern.attribution",   {"pattern_type": "relu"}, mnistutils.heatmap,        "PatternAttribution",            True),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, mnistutils.heatmap,        "DeepTaylor",                    False),
    ("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient",              False),
    ("integrated_gradients",  {"reference_inputs": ri}, mnistutils.heatmap,        "Integrated Gradients",          False),
    # ("deep_lift.wrapper",     {"reference_inputs": ri}, mnistutils.heatmap,        "DeepLIFT Wrapper - Rescale",    False),
    # ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
                                                        # mnistutils.heatmap,        "DeepLIFT Wrapper - RevealCancel", False),
    ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z",                         False),
    ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon",                   False),
]


model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# Create analyzers.
analyzers = []
analyzer_store_path = f"{path_}pretrained/analyzer_%s.npz"
for method in methods:
    analyzer_file = analyzer_store_path % method[0]

    if method[4] and os.path.exists(analyzer_file): # Store analyzer
        print("Loading analyzer: %s" % method[0])
        # TODO restore analyzer
        analyzer = AnalyzerBase.load_npz(analyzer_file)
    else: 
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters

        if method[4]:
            print("Fitting analyzer: %s" % method[0])
            # Some analyzers require training.
            analyzer.fit(data[0], batch_size=256, verbose=1)
            analyzer.save_npz(analyzer_file)
    
    analyzers.append(analyzer)

# # # # # # 
# Do analysis
n = 10
test_images = list(zip(data[2][:n], data[3][:n]))


analysis = np.zeros([len(test_images), len(analyzers)+1, 28, 28, 3])
text = []

os.makedirs(f"{path_}figures", exist_ok=True)
for i, (x, y) in enumerate(test_images):
    np.save(f"{path_}figures/test_image_{i}_{y}.np", x)

for i, (x, y) in enumerate(test_images):
    # Add batch axis.
    x = x[None, :, :, :]

    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x)[0]
    prob = model.predict_on_batch(x)[0]
    y_hat = prob.argmax()

    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output
                 "%s" % label_to_class_name[y_hat] # predicted label
                ))

    for aidx, analyzer in enumerate(analyzers):
        if "deep_lift" in methods[aidx][0]: continue # Deep lift not working properly, so we skip it

        # Analyze.
        a = analyzer.analyze(x)

        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
        a = mnistutils.postprocess(a)
        # Apply analysis postprocessing, e.g., creating a heatmap.
        a = methods[aidx][2](a)
        # Store the analysis.
        analysis[i, aidx] = a[0]


# # # # # # #
# MeLIME
def model_predict(x_e):
    prob = model.predict_on_batch(x_e)
    y_hat = prob[0].argmax()
    return prob

x_train = data[0]
y_train = data[1]

# Training the VAEGen
vae_gen_path = f"{path_}pretrained/vae_gen_mnist_innvestigate1.melime"
generator = VAEGen(input_dim=784, verbose=True, device='cpu')
if os.path.exists(vae_gen_path): 
    generator = generator.load_manifold(vae_gen_path)
else:
    generator.fit(x_train, epochs=20)
    generator.save_manifold(vae_gen_path)


def explanation_melime(x_explain):
    explain_linear = Explainer(
        model_predict=model_predict,
        generator=generator,
        local_model='SGD'
    )
    y_explain = model_predict(x_explain)
    y_explain_index = np.argmax(y_explain)
    explanation, contra = explain_linear.explain_instance(
            x_explain=x_explain,
            r=2.0,
            n_samples=500,
            class_index=y_explain_index,
            tol_importance=0.1,
            tol_error=0.1,
            weight_kernel=None,
            local_mini_batch_max=100,
            scale_data=False
        )
    return explanation.importance

for i, (x, y) in enumerate(test_images):
    a = explanation_melime(x.reshape(-1, 28,28,1)).reshape(1, 28,28,1)
    a = mnistutils.postprocess(a)
    a = mnistutils.heatmap(a)
    analysis[i,-1] =a[0]


# # # # # #
# Plot 
# Prepare the grid as rectengular list
grid = [[analysis[i, j] for j in range(analysis.shape[1])]
        for i in range(analysis.shape[0])]
# Prepare the labels
label, presm, prob, pred = zip(*text)
row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
col_labels = [''.join(method[3]) for method in methods] + ["MeLIME"]

# Plot the analysis.
eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                       file_name=os.environ.get("PLOTFILENAME", None))

print("Done")
