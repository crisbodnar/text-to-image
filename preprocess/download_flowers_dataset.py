from tqdm import tqdm
import requests
import os
import tarfile
from urllib import parse

FLOWERS_DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
FLOWERS_DIR = './data/flowers/'


def download_and_untar(url: str, extract_dir, delete_tar=True):
    response = requests.get(url, stream=True)
    response_size = int(response.headers.get('content-length', 0))

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    filename = parse.urlsplit(url).path.split('/')[-1]

    print('Downloading %s ...' % filename)
    with open(os.path.join(extract_dir, filename), 'wb') as file:
        for data in tqdm(response.iter_content(), total=response_size, unit='B', unit_scale=True, miniters=1):
            file.write(data)

    print('Unzipping %s ...' % filename)
    tar = tarfile.open(os.path.join(extract_dir, filename))
    tar.extractall()
    tar.close()

    if delete_tar:
        os.remove(os.path.join(extract_dir, filename))


def main():
    download_and_untar(FLOWERS_DATASET_URL, FLOWERS_DIR)


if __name__ == '__main__':
    main()
