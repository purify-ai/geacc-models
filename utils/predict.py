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
import datetime
import numpy as np
import pandas as pd
from keras import models
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input

img_size = 224 # img width and height
model = 0
result_columns = ['Filename', 'Prediction', 'Latency (ms)', 'Benign Score', 'Malign Score']

def load_image(img_path):
    img = image.load_img(img_path,
                interpolation = 'lanczos',
                target_size = (img_size, img_size))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = preprocess_input(img_tensor)               # encode to match the expected range of values
    return img_tensor

def predict_image(img_path):
    timer_start = datetime.datetime.now()

    image = load_image(img_path)
    predictions = model.predict(image)
    
    timer_elapsed = int(round((datetime.datetime.now() - timer_start).microseconds / 1000))

    return [img_path, 
            "benign" if predictions[0][0] > predictions[0][1] else "malign", # predicted class with 0.5 cutoff
            timer_elapsed, 
            "{0:.4f}".format(predictions[0][0]), # benign score
            "{0:.4f}".format(predictions[0][1])] # malign score

def main(argv):
    global model
    results = []

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the input image or folder containing images.\
                        Only jpeg images are supported.")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Path to the trained Keras model in HDF5 file format")
    parser.add_argument("-c", "--csv_path", required=False,
                        help="Optional. If specified, results will be saved to CSV file instead of stdout")
    args = parser.parse_args()

    ### Load model
    model = models.load_model(args.model_path)

    if (os.path.isfile(args.input_path)):
        predict_res = predict_image(args.input_path)
        results.append(predict_res)

    elif (os.path.isdir(args.input_path)):
        for f in os.listdir(args.input_path):
            img_path = os.path.join(args.input_path, f)
            if img_path.lower().endswith(('.jpg', '.jpeg', 'png')):
                predict_res = predict_image(img_path)
                results.append(predict_res)

    if args.csv_path:
        pd.DataFrame(results, columns=result_columns).to_csv(args.csv_path, encoding='utf-8', index=False)
    else:
        print(pd.DataFrame(results, columns=result_columns).to_string())

if __name__ == "__main__":
    main(sys.argv)