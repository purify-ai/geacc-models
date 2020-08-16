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

# Evaluate Geacc trained model
# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from geacc.train import HPARAMS
from geacc.training import image_preprocessing
from tf_explain.core.grad_cam import GradCAM
from sklearn import metrics as skmetrics

# Hyperparameters
MODEL_PATH = 'models/geacc-30k/models/Geacc_InceptionV3_1594054432_ep28_vl0.23.tf'
OUTPUT_PATH = 'models/geacc-30k/predicted.csv'


def draw_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    # fig = plt.figure(figsize=figsize)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def draw_pr_curve(precision, recall, class_name):
    data_plot = pd.DataFrame({"Recall": recall, "Precision": precision})
    plt.grid()
    plt.title(f'PR curve for {class_name} class')
    sns.set(style='dark')
    sns.lineplot(x="Recall", y="Precision", data=data_plot)


# TODO: Copied from train_inceptionv3.py
def get_filenames(data_subset):
    """Get TFRecord filenames for dataset.

    Args:
        data_subset: Dataset type "train", "validate" or "test"
    """
    filenames = [
        os.path.join(HPARAMS['dataset_path'], '{}-{:05d}-of-{:05d}'.format(data_subset, i, HPARAMS[f'{data_subset}_tfrecord_files']))
            for i in range(HPARAMS[f'{data_subset}_tfrecord_files'])]

    if HPARAMS['tpu_address'] and not distribution_utils.tpu_compatible_files(filenames):
        raise Exception("TPU requires files stored in Google Cloud Storage (GCS) buckets.")

    return filenames


# %%
# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# %%
OUTPUT_CLASSES_NUM = len(HPARAMS["class_names"])
HPARAMS['dataset_path'] = 'models/geacc-30k/tfrecord'

test_input_dataset = image_preprocessing.input_fn(
    is_training = False,
    filenames   = get_filenames(data_subset="test"),
    batch_size  = HPARAMS['batch_size'],
    num_epochs  = HPARAMS['total_epochs'],
    parse_record_fn = image_preprocessing.get_parse_record_fn(one_hot_encoding_class_num=OUTPUT_CLASSES_NUM),
    dtype=HPARAMS['dtype'],
    drop_remainder=True,
)

steps_test = HPARAMS['test_image_files'] // HPARAMS['batch_size']
print(f"Steps on test: {steps_test}")


# %%
print("Predicting using test dataset...")
predictions = model.predict(
    test_input_dataset,
    steps=steps_test
)

# Calculate true labels
true_labels = []
test_labels = test_input_dataset.map(lambda x, y: tf.dtypes.cast(y, tf.int8))
for tf_labels in test_labels:
    true_labels.extend(tf_labels.numpy())
true_labels = np.array(true_labels)


# %%
# Produce confusion matrix

# Convert the predicted classes from arrays to integers.
pred_class_indices = np.argmax(predictions, axis=1)
true_class_indices = np.argmax(true_labels, axis=1)

# Get the confusion matrix using sklearn.
cfmx = skmetrics.confusion_matrix(y_true=true_class_indices,  # True class for test-set.
                                  y_pred=pred_class_indices)  # Predicted class.

draw_confusion_matrix(cfmx, HPARAMS['class_names'], (4, 3))

# %%
# PR Curves

pr_data_frames = []
for i in range(OUTPUT_CLASSES_NUM):
    #plt.subplots(i, 1)
    class_true_labels = true_labels[:, i]
    class_predictions = predictions[:, i]
    precision, recall, thresholds = skmetrics.precision_recall_curve(class_true_labels, class_predictions)
    class_average_precision = skmetrics.average_precision_score(class_true_labels, class_predictions)
    pr_data_frames.append(pd.DataFrame({"Recall": recall, "Precision": precision,
                                        "Class": f'{HPARAMS["class_names"][i]} class (ap: {class_average_precision:.3f})'}))

# Micro-averaged precision over all classes
micro_precision, micro_recall, _ = skmetrics.precision_recall_curve(true_labels.ravel(), predictions.ravel())
micro_average_precision = skmetrics.average_precision_score(true_labels, predictions, average="micro")
pr_data_frames.append(pd.DataFrame({"Recall": micro_recall, "Precision": micro_precision,
                                    "Class": f'micro-averaged over all classes (ap: {micro_average_precision:.3f})'}))

pr_data_plot = pd.concat(pr_data_frames)

plt.figure(figsize=(15,14))
plt.grid()
plt.title(f'PR curves')
sns.set_style("dark", {"xtick.major.size": 16, "ytick.major.size": 16})
sns.lineplot(x="Recall", y="Precision", hue="Class", data=pr_data_plot)


# %%
# Explain images which were misclassified
explainer = GradCAM()

diff_class_indices = true_class_indices - pred_class_indices
false_indeces = np.argwhere(diff_class_indices != 0)
test_input_dataset_unbatched = test_input_dataset.unbatch() #.map(lambda x, y: ([x], y))
i = 0
for test_record in test_input_dataset_unbatched:
    if diff_class_indices[i] != 0:
        test_record_label = np.argmax(test_record[1].numpy())
        print(f"Image #{i}: Record: {test_record_label} True: {true_class_indices[i]} Predicted: {pred_class_indices[i]}")

        img_data = ([test_record[0].numpy()], test_record[1].numpy())
        grid = explainer.explain(img_data, model, class_index=pred_class_indices[i])
        explainer.save(grid, "models/geacc-30k/explain/", f"{i}-t{true_class_indices[i]}-p{pred_class_indices[i]}.png")
    i += 1

# %%
# Detailed list of all filenames, predictions and scores
labels = (test_img_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions_labeled = [labels[k] for k in pred_class_indices]

pd.set_option('display.max_rows', None)
cdf = pd.DataFrame({"Filename": test_img_generator.filenames[:len(test_classes)],
                    "Prediction": predictions_labeled,
                    "Benign Score": ["{0:.4f}".format(i[0]) for i in predictions],
                    "Malign Score": ["{0:.4f}".format(i[1]) for i in predictions]})

cdf.to_csv(output_path, encoding='utf-8', index=False)
