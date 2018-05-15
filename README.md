# Fast neural style with MobileNetV2 bottleneck blocks
This repository contains a PyTorch implementation of an algorithm for artistic style transfer. The implementation is based on the following papers and repositories:

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [PyTorch Fast Neural Style Example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

## Main Differences from Other Implementations

* Residual Blocks and Convolutions are changed to MobileNetV2 bottleneck blocks which make use of Inverted Residuals and Depthwise Separable Convolutions.

  ![Bottleneck](https://hsto.org/webt/wl/yo/sz/wlyoszqnws58itd4ojt1cqt7sng.png)

  On the picture you can see 2 types of MobileNetV2 bottleneck blocks. Left one is used instead of residual block and right one is used instead of convolution layer. Purposes of this change:

  - Decrease number of trainable parameters of the transformer network from __~1.67m__ to __~0.23m__, therefore decrease amount of the memory used by the transformer network.

  - In theory this should give a good speedup during training time and, more importantly, during inference time (fast neural style should be fast as possible). It appeared that in practice things are not so good and this architecture of the transformer network is only a bit faster than the original transformer network. The main cause of it is that depthwise convolutions are not so efficiently implemented on GPU as common convolutions are (on CPU the speedup is bigger, but still not drastic).

* This implementation uses the feature extractor wrapper around PyTorch module which uses PyTorch hook methods to retrieve layer activations. With this extractor:

  - You don't need to write a new module wrapper in order to extract desired features every time you want to use a new loss network. You just need to input model and layer indexes to the feature extractor wrapper and it will handle extracting for you. (__Note__: The wrapper flattens the input module/model so you need to input proper indexes of the flattened module, i.e. if module/model is a composition of smaller modules it will be represented as flat list of layers inside the wrapper).

  - Makes training process slightly faster.

* The implementation allows you use different weights for different style features, which leads to better visual results.

## Requirements
- [pytorch](https://pytorch.org) (>= 0.4.0)
- [torchvision](https://pytorch.org)
- [PIL](https://pillow.readthedocs.io/en/5.1.x/)
- [OpenCV](https://opencv.org/) (for webcam demo)
- GPU is not necessary

## Usage
To train the transformer network:
```
python fnst.py -train
```
To stylize an image with a pretrained model:
```
python fnst.py
```

All configurable parameters are stored as globals in the top of `fnst.py` file, so in order to configure those parameters just change them in `fnst.py` (I thought it is more convenient way than adding dozen of arguments).

There is also webcam demo in the repo, to run it:
```
python webcam.py
```
`webcam.py` also has some globals in the top of the file which you can change.


## Examples
All models were trained on 128x128 (because of GTX 960m on my laptop) [COCO Dataset](http://cocodataset.org) images for 3 epochs.

- Styles

  <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/mosaic.jpg" width="256" height="256"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/picasso.jpg" width="256" height="256"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/candy.jpg" width="256" height="256">
- Results

  <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/dancing.jpg" width="300"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/dancing_mosaic.jpg" width="300">
  <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/dancing_picasso.jpg" width="300"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/dancing_candy.jpg" width="300">

  <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/pwr.jpg" width="400"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/pwr_mosaic.jpg" width="400">
  <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/pwr_picasso.jpg" width="400"> <img src="https://raw.githubusercontent.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/master/images/results/pwr_candy.jpg" width="400">
