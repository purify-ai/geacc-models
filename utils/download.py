#!/usr/bin/env python

# Copyright 2019 Purify Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import argparse
import logging

import requests
import urllib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import pandas as pd
import dask
from dask.diagnostics import ProgressBar

from PIL import Image
from io import BytesIO
import ratelim
from checkpoints import checkpoints

class_names = {0: 'benign', 1: 'explicit', 2: 'suggestive'}
download_path = 'dataset'
request_timeout = 120
img_optimise_size = 0
download_threads = 10
overwrite_existing = False
logfile = 'download.log'

dask.config.set(scheduler='threads', num_workers=download_threads)

# create logger with 'dataset_download'
logger = logging.getLogger('dataset_download')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(logfile)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


def init_folders():
    # Write the images to files, adding them to the package as we go along.
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    for class_id, class_name in class_names.items():
        if not os.path.isdir(f"{download_path}/{class_name}"):
            os.mkdir(f"{download_path}/{class_name}")


def download(metadata_url):
    """Download images from origins"""

    # Download or load the image metadata pandas DataFrame
    metadata = pd.read_csv(metadata_url)

    # Start from prior results if they exist and are specified, otherwise start from scratch.
    remaining_todo = len(metadata.index) if checkpoints.results is None else len(
        metadata.index) - len(checkpoints.results)
    print(f"Downloading {len(metadata.index)} images"
          f" ({len(metadata.index) - remaining_todo} have already been downloaded)")

    # Download the images
    delayed_futures = []

    for index, row in metadata.iterrows():
        delayed_futures.append(dask.delayed(_download_image)(row))

    with ProgressBar():
        dask.compute(*delayed_futures)


#@ratelim.patient(5, 5)
def _download_image(row):
    """Download a single image from a URL, rate-limited to once per second"""
    image_url = row['OriginalURL']

    try:
        class_name = class_names[row['ClassID']]
        image_name = _generate_filename(image_url)
        filename = f"{download_path}/{class_name}/{image_name}"

        # skip if file exists and we do not overwrite
        if os.path.isfile(filename) and not overwrite_existing:
            return

        s = requests.Session()
        retries = Retry(total=8, connect=4, read=4, redirect=4,
                        backoff_factor=2, status_forcelist=[400, 422, 500, 502, 503, 504])
        s.mount('http://', HTTPAdapter(max_retries=retries))
        s.mount('https://', HTTPAdapter(max_retries=retries))
        response = s.get(image_url, timeout=request_timeout)

        if response.ok:
            if img_optimise_size == 0:
                with open(filename, "wb") as f:
                    f.write(response.content)
            else:
                _image_resize_and_save(response.content, filename)

        response.raise_for_status()
    except ValueError as e:
        logger.error(f"Cannot download {image_url} due to the input value: {e}")
    except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
        logger.error(f"Download failed for {image_url} with message: {e}")


def _generate_filename(url):
    """Use encoded URL as a filename. This helps with dataset troubleshooting. 
    At any point it is possible to URL decode filename and obtain original url.

    Not all URLs can encoded with this scheme as filename length restricted 
    to 250 symbols on many filesystems.
    """
    filename = urllib.parse.quote_plus(url)

    if len(filename) >= 250:
        raise ValueError(f"File name too long: {filename}")

    return filename


def _image_resize_and_save(img_content, filename):
    orig_content = BytesIO(img_content)
    
    try:
        img = Image.open(orig_content)

        # convert from other modes
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # create thumbnail no smaller than the given size
        width, height = img.size
        r = max(img_optimise_size / width, img_optimise_size / height)
        new_size = width * r, height * r
        img.thumbnail(new_size, resample=Image.LANCZOS)

        # make sure that the resized image has smaller size, otherwise keep the original
        new_content = BytesIO()
        img.save(new_content, format='JPEG', optimize=True, quality=80)
        orig_size = orig_content.tell()
        new_size = new_content.tell()

        if (new_size < orig_size):
            img.save(filename, format='JPEG', optimize=True, quality=80)
        else:
            with open(filename, "wb") as f:
                f.write(img_content)
    except Exception as e:
        logger.info(f"Image resizing failed for {filename} with message: {e}. Saving original instead.")
        with open(filename, "wb") as f:
            f.write(img_content)


def main(argv):
    global download_path
    global img_optimise_size
    global overwrite_existing
    global download_threads

    parser = argparse.ArgumentParser()
    parser.add_argument("download_path", default="dataset",
                        help="Folder where downloaded dataset images will be stored.")
    parser.add_argument("-m", "--metadata", metavar="CSV_FILE", required=True,
                        help="Local or online CSV file containing dataset metadata.")
    parser.add_argument("-o", "--overwrite", required=False, default=False, action="store_true",
                        help="Overwrite if file already exists. By default existing are not downloaded/overwritten.")
    parser.add_argument("-w", "--workers", metavar="WORKERS", required=False, type=int, default=10,
                        help="Number of concurrent workers used for downloading. Use to adjust the throughput. Default=10.")
    parser.add_argument("-r", "--resize", metavar="SIZE", required=False, type=int, default=0,
                        help="Resize large images to on the fly to reduce required storage. Output images edges will be no smaller than SIZE px and saved as optimised JPEG(80).")

    args = parser.parse_args()

    download_path = args.download_path
    img_optimise_size = args.resize
    overwrite_existing = args.overwrite

    init_folders()
    download(args.metadata)


if __name__ == '__main__':
    main(sys.argv)
