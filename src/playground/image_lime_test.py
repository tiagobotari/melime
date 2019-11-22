import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import os, json, sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import models, transforms

from density_lime import lime_image
from density_lime.gan_density import GANDensity
from skimage.segmentation import mark_boundaries

# # # # # # # # # # Imagenet class indexes  - Not used that much currently but might be important later.
def imagenet_indexes():
    idx2label, cls2label, cls2idx = [], {}, {}
    with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
        class_idx   = json.load(read_file)
        idx2label   = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label   = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        cls2idx     = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
    return idx2label, cls2label, cls2idx
# # # # # # # # # #

def get_image(path):
    """
        Load image path into a PIL.Image object.
    """
    with open(os.path.abspath(path), 'rb') as f: 
        with Image.open(f) as img:
            return img.convert('RGB')

# # # # # # # # # # # # # Torch helper functions
# resize and take the center crop part of image to what our model expects

# Standard normalization constants for ImageNet networks from torchvision.models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def get_input_transform():
    """
        Conver PIL.Image to a torch.Tensor with ~mean 0 and std 1.
        This will be used, when data is passed to the classifiers.
    """
    transf = transforms.Compose([
        transforms.Resize((256, 256)),  # Works on PIL images
        transforms.CenterCrop(224),     # Works on PIL images
        transforms.ToTensor(),          # Converts 0..255 to floats of 0..1
        normalize                       # Normalize data to have ~mean 0 and std 1
    ])
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0) # convert image to a 1, H, W, C torch.Tensor

def get_pil_transform():
    """
        First half of transform in `get_input_transform` 
        Keeps data in range 0..255.
        This will be used when data is passed to LIME.
    """
    transf = transforms.Compose([
        transforms.Resize((256, 256)),  # Works on PIL images
        transforms.CenterCrop(224),     # Works on PIL images
    ])
    return transf

def get_preprocess_transform():
    """
        Second half of transform in `get_input_transform`.
        Takes data from 0..255 to ~mean 0 and std 1.
        This is used to transform images before running them through
        the black box (`model`) during explanation processiong.
    """
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf


def main(args):
    """
        Runs LIME with GAN density sampling
    """

    pill_transf = get_pil_transform()                   # First half of `get_input_transform`
    preprocess_transform = get_preprocess_transform()   # Second half of `get_input_transform`

    # CUDA if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare model
    model = models.resnet101(pretrained=True)
    model.to(device)
    model.eval()

    mean_explainer = lime_image.DensityImageExplainer(None)
    density = GANDensity()
    gan_explainer = lime_image.DensityImageExplainer(density)

    explainers = [("Mean", mean_explainer), ("GAN", gan_explainer)]

    # Prepare prediction function for explainer to use to get labels
    def batch_predict(images):
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0).to(device)
        logits = model(batch)

        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


    # Prepare input
    if args.image is not None and len(args.image) > 0:
        names   = args.image
        images  = [get_image('./data/%s.jpg' % a) for a in args.image]
    else:
        names   = ['dog-cat']
        images  = [get_image('./data/dogs.png')]

    images_t = [get_input_tensors(img).to(device) for img in images]

    idx2label, *_ = imagenet_indexes()

    fig, ax = plt.subplots(max(2, len(names)), 1 + len(explainers), figsize=(9, 3 * len(names)))
    for i, (name, img, img_t) in enumerate(zip(names, images, images_t)):

        # Predict
        pred = torch.argmax(model(img_t)).item()

        print("max logit: ", pred, "Class: ", idx2label[pred])
        np_img = np.array(pill_transf(img))

        # ax[i, 0].set_title("N: %s | P: %s" % (name, idx2label[pred]))
        def remove_axis(ax):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        ax[i, 0].imshow(np_img)
        remove_axis(ax[i, 0])

        # Create explanations and plot them.
        for j, (ex_name, explainer) in enumerate(explainers): 
            explanation = explainer.explain_instance(np_img,
                                                     batch_predict, # classification function
                                                     top_labels=5,
                                                     hide_color=None,
                                                     num_samples=1000) # number of images that will be sent to classification function

            temp, mask  = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=255)
            boundary    = mark_boundaries(temp/255.0, mask)

            print(ex_name, temp.dtype, temp.min(), temp.max(), temp.shape)
            
            ax[i, j+1].imshow(temp.astype(np.uint8))
            remove_axis(ax[i, j+1])

            if i == 0: 
                ax[i, j+1].set_title(ex_name)

    fig.tight_layout()
    fig.savefig("plots/GAN_explanations_%s.pdf" % "_".join(names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Explainer")

    parser.add_argument("--image", '-i', type=str, nargs="+")

    args = parser.parse_args()
    main(args)





