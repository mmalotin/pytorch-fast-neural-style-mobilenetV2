import torch
import torch.nn.functional as F
from PIL import Image


def load_im(f, size=None):
    img = Image.open(f)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


def save_im(f, tens):
    img = tens.detach().clamp(0, 255).cpu().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(f)


def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c*h*w)
    return gram


def norm_batch(b):
    mean = b.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = b.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    b = b.div_(255.0)
    return (b - mean) / std


def regularization_loss(x):
    loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
    return loss
