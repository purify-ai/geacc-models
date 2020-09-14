#!/usr/bin/env python3

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
import datetime
import pandas as pd

from geacc.training.image_preprocessing import preprocess_image
import tensorflow as tf

INPUT_IMG_SIZE = 299  # img width and height
NUM_CHANNELS = 3
model = None
result_columns = ['Filename', 'Latency (ms)', 'Explicit Only', 'Explicit and Suggestive']


def _process_image(filename):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    if not tf.io.is_jpeg(image_data):
        raise Exception("Unsupported image type. Skipping.")

    image = preprocess_image(
        image_buffer=image_data,
        output_height=INPUT_IMG_SIZE,
        output_width=INPUT_IMG_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=False)

    return image


def predict_image(img_path):
    timer_start = datetime.datetime.now()

    image = _process_image(img_path)
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)

    timer_elapsed = int(round((datetime.datetime.now() - timer_start).microseconds / 1000))

    return [img_path,
            timer_elapsed,
            "%.4f" % predictions[0][0],  # explicit only
            "%.4f" % predictions[1][0]]  # explicit and suggestive


def main(argv):
    global model
    results = []

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the input image or folder containing images.")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Path to the trained Keras model in HDF5 file format")
    parser.add_argument("-c", "--csv_path", required=False,
                        help="Optional. If specified, results will be saved to CSV file instead of stdout")
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    if (os.path.isfile(args.input_path)):
        predict_res = predict_image(args.input_path)
        results.append(predict_res)

    elif (os.path.isdir(args.input_path)):
        for f in os.listdir(args.input_path):
            img_path = os.path.join(args.input_path, f)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tif', '.tiff')):
                predict_res = predict_image(img_path)
                results.append(predict_res)

    if args.csv_path:
        pd.DataFrame(results, columns=result_columns).to_csv(args.csv_path, encoding='utf-8', index=False)
    else:
        print(pd.DataFrame(results, columns=result_columns).to_string())


if __name__ == "__main__":
    main(sys.argv)
