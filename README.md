# Geacc: General Audience Content Classifier
GEneral Audience Content Classifier (Geacc) is a pre-trained deep neural network model for classifying images which are suitable for general audiences with a particular focus on children and adolescents below the age of 12.

Currently, the model detects sexually explicit content. In the future, we plan to extend it to other types of harmful visual content such as violence and horror.

Please note that Geacc is not designed to be a general purpose porn classifier. It is deliberately trained to be stricter. We like to think of it as a parent of 8 years old, although even that would be a very subjective criterion.

To minimise the number of false-positives in this particular context, for "benign" training images emphasis was made on the scenes which children come across most often: cartoons, children movies, toys, games, nurseries, playgrounds, etc.

If you find it useful, please let us know about your use case by filling in short form here. As a non-profit organisation, it's essential for us to gauge the impact of our work.

## Download
Pre-trained model is available in Keras HDF5 format and can be [downloaded here](https://github.com/purify-ai/geacc-models/releases).

## Definitions
Definition of "General Audience" varies depending on the country and type of content. Our definition influenced by [Television Content Rating systems](https://en.wikipedia.org/wiki/Television_content_rating_system) which are stricter than movie and gaming rating systems. TV content rating systems of many countries define "General Audience" content suitable for children under 12.

The precise definition of sexually explicit content is also highly subjective. In this project we consider _any visual material that may cause sexual arousal or fantasy, whether intentional or unintentional_, to be harmful. Framing the problem in that particular way prioritises child safety over objectivity, removes ambiguity and makes machine learning model more robust.

## Deep Neural Network architecture
This model based on lightweight [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) architecture which was specifically designed to run on personal mobile devices.

This choice aligned with our vision for an on-device child protection systems which we [described here](https://medium.com/purify-foundation/how-artificial-intelligence-can-help-protect-children-37ce51b75c35).

In the future, we plan to release pre-trained models using other architectures which may provide higher accuracy for those who don't need to run inference on mobile devices (e.g. Inception).

## Data set
The dataset consists of 50.000 images, equally split between two classes - _benign_ and _malign_. In addition to that, test dataset of ~3.600 images used for validation.

| Class    | Training Images | Test Images |
| -------- | ------- | ------ |
| Benign   | 25,000  | 1,800  |
| Malign   | 25,000  | 1,800  |

Images in this dataset are mainly collected online, while some images taken from [Caltech256](https://authors.library.caltech.edu/7694/) and [Porn Database](https://sites.google.com/site/pornographydatabase/).

We do not provide original dataset due to the nature of the data and the fact that Purify Foundation does not own the copyright of those images. However, in the future, we plan to publish the list of URLs.

## Training Process
Training was performed using Keras with TensorFlow backend.

Instead of training from scratch, we used fine-tuning approach. MobileNetV2 model pre-trained with ImageNet was used as the basis for the Geacc model. The top fully connected layer was replaced for binary classification (malign/benign). All layers except the top 30 layers were frozen.

Images were resized and cropped to match 224x224 input size and augmented to improve accuracy.

Hyperparameters and other details can be found in the source code of the training script.

## Results
Geacc model achieved more than 95% accuracy on the test dataset. Confusion matrix below has a more granular view of the results (`cutoff=0.5`).

![alt text](assets/geacc-cm.png?raw=true "Geacc Results Confusion Matrix")

## How does it compare to Yahoo! OpenNSFW?
For comparison, we also ran our test data though OpenNSFW model with `cutoff=0.5`. As can be seen in the confusion matrix below, OpenNSFW model is less strict with malign images.

![alt text](assets/opennsfw-cm.png?raw=true "OpenNSFW Results Confusion Matrix")

## Performing Classification

To classify images using Geacc model, you can use `utils/predict.py` script provided in this repo. Note that it requires Keras and TensorFlow dependencies.

Usage:

```
% python3 utils/predict.py -h

usage: predict.py [-h] -m MODEL_PATH [-c CSV_PATH] input_path

positional arguments:
  input_path            Path to the input image or folder containing images.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the trained Keras model in HDF5 file format
  -c CSV_PATH, --csv_path CSV_PATH
                        Optional. If specified, results will be saved to CSV
                        file instead of stdout
```

Example:

```
% python3 utils/predict.py -m ../models/PurifyAI_Geacc_MobileNetV2_224.h5 ../img/

Using TensorFlow backend.
            Filename Prediction  Latency (ms) Benign Score Malign Score
0    ../img/dog1.jpg     benign            62       0.7445       0.2555
1   ../img/nsfw1.jpg     malign            89       0.0000       1.0000
2   ../img/nsfw2.jpg     malign            91       0.0150       0.9850
3    ../img/cat1.jpg     benign            74       0.8820       0.1180
```

## Disclaimer
This project is currently in the early development stage. We do not provide guarantees of output accuracy.

## License
Models and source code are licensed under [Apache License 2.0](LICENSE)