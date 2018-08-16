# Text to Image Synthesis using Generative Adversarial Networks

This is the official implementation for my thesis on  [Text to Image Synthesis using Generative Adversarial Networks](https://arxiv.org/abs/1805.00676). Please be aware that the code is in an experimental stage and it might require some small tweaks.

## How to download the dataset

1. Setup your `PYTHONPATH` to point to the root directory of the project.
2. Download the preprocessed [flowers text descriptions](https://drive.google.com/file/d/0B3y_msrWZaXLaUc0UXpmcnhaVmM/view) 
and extract them in the `/data` directory.
3. Download the [images from Oxford102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) 
and extract the images in `/data/flowers/jpg`. You can alternatively run `python prep_incep_img/download_flowers_dataset.py` from the 
root directory of the project.
4. Run the `python prep_incep_img/preprocess_flowers.py` script from the root directory of the project.

### Requirements

- python 3.6
- tensorflow 1.4
- scipy
- numpy
- pillow
- easydict
- imageio
- pyyaml

