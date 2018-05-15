import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from PIL import Image
from tqdm import tqdm
import os
import argparse

from utils import load_im, save_im
from utils import gram_matrix, norm_batch, regularization_loss
from fnst_modules import TransformerMobileNet
from feature_ext import FeatureExtractor

EPOCHS = 2
LOSS_NETWORK = models.vgg16
TRAIN_PATH = '101_ObjectCategories'
STYLE_IMAGE = 'images/mosaic.jpg'
CHECK_IMAGE = 'images/dancing.jpg'
IMAGE_SIZE = 128
LAYER_IDXS = [3, 8, 15, 22]
BATCH_SIZE = 4
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 3 * 1e5
REG_WEIGHT = 3 * 1e-5
STYLE_PROPORTIONS = [.35, .35, .15, .15]
CONTENT_INDEX = 1
LOG_INTERVAL = 1000
CHECKPOINT = 4000

OUTPUT_PATH = 'images/results'
INPUT_IMAGE = 'images/pwr.jpg'
MODEL_PATH = 'models/mosaic.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Learner():
    def __init__(self, loss_network, train_path, style_image, check_image, im_size,
                 layer_idxs, batch_size, c_weight, s_weight, r_weight,
                 style_proportions, content_index, log_interval, checkpoint):
        assert len(style_proportions) == len(layer_idxs)

        # prepare dataset and loader
        self.dataset, self.loader = (
            self.__prepare_dataset(train_path, im_size, batch_size))

        # prepare transformer and classifier nets, optimizer
        self.tfm_net = TransformerMobileNet().to(device)
        self.loss_net = loss_network
        self.__prepare_loss_net()
        self.optimizer = Adam(self.tfm_net.parameters(), 1e-3)

        # prepare feature extractor, style image
        # and image for checking intermediate results
        self.content_index = content_index
        self.fx = FeatureExtractor(self.loss_net, layer_idxs)
        self.style_batch, self.style_target = (
            self.__prepare_style_target(style_image, batch_size))
        self.check_tensor = self.__prepare_check_tensor(check_image)

        # set weights for different losses
        self.content_weight = c_weight
        self.style_weights = [s_weight*x for x in style_proportions]
        self.reg_weight = r_weight

        # intervals
        self.log_intl = log_interval
        self.checkpoint = checkpoint

    def __prepare_paths(self):
        for p in ['models', 'images']:
            _path = os.path.join('tmp', p)
            os.makedirs(_path, exist_ok=True)

    def __prepare_dataset(self, train_path, im_size, batch_size):
        transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))])

        ds = datasets.ImageFolder(train_path, transform)
        ld = DataLoader(ds, batch_size=batch_size)
        return ds, ld

    def __prepare_loss_net(self):
        for p in self.loss_net.parameters():
            p.requires_grad_(False)
        self.loss_net.to(device).eval()

    def __prepare_style_target(self, im_path, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))])

        style_im = load_im(im_path)
        style_tensor = transform(style_im)
        style_batch = style_tensor.repeat(batch_size, 1, 1, 1).to(device)

        self.loss_net(norm_batch(style_batch))
        style_target = tuple(gram_matrix(x)
                             for x in self.fx.features)
        return style_batch, style_target

    def __prepare_check_tensor(self, im_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))])

        if im_path is None:
            res = self.dataset[0].unsqueeze(0).to(device)
        else:
            res = transform(Image.open(im_path)).unsqueeze(0).to(device)
        return res

    def train(self, epochs):
        self.__prepare_paths()
        for e in range(epochs):
            self.tfm_net.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            agg_reg_loss = 0.
            for i, (x, _) in enumerate(tqdm(self.loader, desc=f'Epoch {e}')):
                len_batch = len(x)
                self.optimizer.zero_grad()

                x = x.to(device)
                y = self.tfm_net(x)
                x = norm_batch(x)
                y = norm_batch(y)

                self.loss_net(y)
                style_y = tuple(gram_matrix(x) for x in self.fx.features)
                content_y = self.fx.features[self.content_index]

                self.loss_net(x)
                content_x = self.fx.features[self.content_index]

                content_loss = F.mse_loss(content_y, content_x)
                content_loss *= self.content_weight

                style_loss = 0.
                for gm_y, gm_t, w in zip(style_y, self.style_target,
                                   self.style_weights):
                    style_loss += (w*F.mse_loss(gm_y, gm_t[:len_batch, :, :]))

                reg_loss = self.reg_weight * regularization_loss(y)

                total_loss = content_loss + style_loss + reg_loss

                total_loss.backward()
                self.optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()
                agg_reg_loss += reg_loss.item()

                if (i+1) % self.log_intl == 0:
                    self.intermediate_res(agg_content_loss, agg_style_loss,
                                          agg_reg_loss, i+1)

                if (i+1) % self.checkpoint == 0:
                    self.save_tfm_net(e+1, i+1)

            self.save_tfm_net(e+1, i+1)

    def intermediate_res(self, c_loss, s_loss, r_loss, n):
        self.tfm_net.eval()
        check = self.tfm_net(self.check_tensor)
        _path = os.path.join('tmp', 'images', f'check{n}.jpg')
        save_im(_path, check[0])
        self.tfm_net.train()

        msg = (f'\nbatch: {n}\t'
               f'content: {c_loss/n}\t'
               f'style: {s_loss/n}\t'
               f'reg: {r_loss/n}\t'
               f'total: {(c_loss + s_loss + r_loss)/n} \n')

        print(msg)

    def save_tfm_net(self, e, i):
        name = f'epoch{e}_batch{i}.pth'
        _path = os.path.join('tmp', 'models', name)
        torch.save(self.tfm_net.state_dict(), _path)


class Stylizer():
    def __init__(self, model_path, output_path):
        self.output_path = output_path
        self.model_name = os.path.basename(model_path).split('.')[0]
        self.__load_net(model_path)
        self.net.to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255).unsqueeze(0).to(device))])

    def __load_net(self, model_path):
        with torch.no_grad():
            self.net = TransformerMobileNet()
            state_dict = torch.load(model_path)
            self.net.load_state_dict(state_dict)

    def stylize(self, im_path):
        with torch.no_grad():
            self.net.eval()
            im = load_im(im_path)
            x = self.transform(im)
            out = self.net(x)
        _name = (os.path.basename(im_path).split('.')[0] + '_'
                 + self.model_name + '.jpg')
        _path = os.path.join(self.output_path, _name)
        save_im(_path, out[0])


parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')


def main():
    args = parser.parse_args()
    if args.train:
        loss_network = LOSS_NETWORK(True)
        loss_network = nn.Sequential(*list(loss_network.features)[:23]) # ToDo
        lrn = Learner(loss_network=loss_network,
                      train_path=TRAIN_PATH, style_image=STYLE_IMAGE,
                      check_image=CHECK_IMAGE, im_size=IMAGE_SIZE,
                      layer_idxs=LAYER_IDXS, batch_size=BATCH_SIZE,
                      c_weight=CONTENT_WEIGHT, s_weight=STYLE_WEIGHT,
                      r_weight=REG_WEIGHT, style_proportions=STYLE_PROPORTIONS,
                      content_index=CONTENT_INDEX, log_interval=LOG_INTERVAL,
                      checkpoint=CHECKPOINT)
        lrn.train(EPOCHS)
    else:
        stl = Stylizer(MODEL_PATH, OUTPUT_PATH)
        stl.stylize(INPUT_IMAGE)


if __name__ == '__main__':
    main()
