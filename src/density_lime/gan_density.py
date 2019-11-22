import sys
import matplotlib.pyplot as plt
import torch
import numpy as np

# # # # # # LOAD GAN CODE # # # # # # # # #
# TODO make proper load of module
sys.path.append('../external_modules/inpainting')
from trainer import Trainer # /generative-inpainting-pytorch'
from utils.tools import get_config
# # # # # # DONE LOADING  # # # # # # # # #

""" 
    Small helperfunctions to simplify other parts of the code
"""
def load_weights(path, device):
    model_weights = torch.load(path)
    return {
        k: v.to(device)
        for k, v in model_weights.items()
    }

to_np       = lambda x: x if not isinstance(x, torch.Tensor) else x.cpu().detach().numpy()
upcast      = lambda x: np.clip((x + 1) * 127.5 , 0, 255).astype(np.uint8)
normalize   = lambda x: np.clip((x - x.min()) / x.max() * 255, 0, 255).astype(np.uint8)

def store_fill(img, fill, mask, i):
    """
        Simple function for storing/plotting fills. 
        Most for debugging and assessment of the method.
    """

    img     = to_np(img)
    fill    = to_np(fill)
    mask    = to_np(mask).squeeze()

    fig, ax = plt.subplots(2, 2)

    ax[0,0].set_title("Image")
    ax[0,0].imshow(upcast(img))
    
    ax[0,1].set_title("Mask")
    ax[0,1].imshow(mask)

    ax[1,0].set_title("(x+1) / 2")
    ax[1,0].imshow(upcast(fill))

    ax[1,1].set_title("(x - x.min) / x.max")
    ax[1,1].imshow(normalize(fill))

    fig.tight_layout()

    if not os.path.exists('./data/fill'): os.mkdir('./data/fill')
    fig.savefig('data/fill/fill_%d.pdf' % i)
    plt.close(fig)


################################################################################# 
# Actual implementation of GAN density                                          #
#################################################################################
class GANDensity:
    def fill(self, image, segments):
        # TODO fix config loading
        config = get_config('../external_modules/inpainting/configs/config.yaml')
        if config['cuda']:
            device = torch.device("cuda:{}".format(config['gpu_ids'][0]))
        else:
            device = torch.device("cpu")

        trainer = Trainer(config)
        trainer.load_state_dict(load_weights('../../pretrained_models/torch_contextual_attention_model.p', device), strict=False)
        trainer.eval()

        img     = torch.tensor(image, device=device).float()
        img     = img.permute(2, 0, 1)                        
        fill    = torch.zeros(img.size(), device=device)      

        for seg in range(segments.max()):
            sel         = torch.tensor(segments, device=device) == seg

            mask        = torch.zeros(segments.shape, device=device).float().unsqueeze(0) 
            mask[:,sel] = 1.
            mask        = mask.unsqueeze(0)

            x           = ( (img / 127.5 - 1) * (1 - mask) ) 

            with torch.no_grad():
                _, res, _ = trainer.netG(x, mask)
                # store_fill(x.squeeze().permute(1, 2, 0), res.squeeze().permute(1, 2, 0), mask, seg)

            fill[:, sel] = res[0,:, sel]

        fill = fill.permute(1, 2, 0).cpu().detach().numpy()
        fill = (fill + 1.) / 2.

        if False: # set to true if you wish to see fill
            fig, ax = plt.subplots()
            ax.imshow(fill - fill.min() / fill.max())
            fig.savefig('fill.pdf')

        return upcast(fill)

