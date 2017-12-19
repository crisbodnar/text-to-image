from tqdm import tqdm
import requests
import os
import tarfile

FLOWERS_DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
FLOWERS_DATASET_TAR_PATH = './data/flowers.tgz'
FLOWERS_DATASET_PATH = './data'


def download_dataset():
    response = requests.get(FLOWERS_DATASET_URL, stream=True)
    response_size = int(response.headers.get('content-length', 0))

    if not os.path.exists(os.path.dirname(FLOWERS_DATASET_TAR_PATH)):
        os.makedirs(os.path.dirname(FLOWERS_DATASET_TAR_PATH))

    print('Downloading the dataset...')
    with open(FLOWERS_DATASET_TAR_PATH, 'wb') as file:
        for data in tqdm(response.iter_content(32*1024), total=response_size, unit='B', unit_scale=True, miniters=1):
            file.write(data)


def untar_dataset():
    with tarfile.open(FLOWERS_DATASET_TAR_PATH) as tar:
        tar.extractall()


def main():
    download_dataset()
    untar_dataset()


if __name__ == '__main__':
    main()
