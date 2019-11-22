import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f: 
        with Image.open(f) as img:
            return img.convert('RGB')

# # # # # # # # # # # # # Torch helper functions
# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # lambda x: x / 255.,
        normalize
    ])
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

# # # # # # # # # #

# # # # # # # # # # Imagenet class indexes
idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
    class_idx   = json.load(read_file)
    idx2label   = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label   = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx     = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
# # # # # # # # # #

# Prepare model
model = models.resnet101(pretrained=True)
model.eval()

# Prepare input
if len(sys.argv) > 1:
    img_src = './data/%s.jpg' % sys.argv[1]
else:
    img_src = './data/dogs.png'

img = get_image(img_src)
img_t = get_input_tensors(img)

# Predict
pred = torch.argmax(model(img_t)).item()
print("max logit: ", pred, "Class: ", idx2label[pred])

# Predictor function for explanation generations
def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

density = GANDensity()
# density = None
explainer = lime_image.DensityImageExplainer(density)
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict, # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.savefig('plot.pdf')
# plt.show()
