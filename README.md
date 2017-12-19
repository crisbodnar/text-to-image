# Text to Photo Realistic Image Synthesis

This project combines in its implementation multiple research papers for text to image synthesys. 
The currently supported dataset is Oxford 102 flowers dataset. 

## How to download the dataset

1. Download the preprocessed [flowers text descriptions](https://drive.google.com/file/d/0B3y_msrWZaXLaUc0UXpmcnhaVmM/view) 
and extract them in the `/data` directory.
2. Download the [images from Oxford102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) 
and extract them in `/data/flowers`. You can alternatively run `python preprocess/download_flowers_dataset.py` from the 
root directory of the project.
3. Run the `python preprocess/preprocess_flowers.py` script from the root directory of the project.

### Requirements

- python 3.6
- tensorflow 1.4
- scipy
- numpy
