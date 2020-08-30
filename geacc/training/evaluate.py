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
    parse_record_fn = image_preprocessing.get_parse_record_fn(
        one_hot_encoding_class_num=OUTPUT_CLASSES_NUM, binarize_label_names=HPARAMS['binarized_label_names']),
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

# Multi-output predictions
predictions_explicit_only = predictions[0]
predictions_explicit_and_suggestive = predictions[1]

# True labels for test dataset
test_labels = list(test_input_dataset.batch(HPARAMS['test_image_files']).as_numpy_iterator())
test_labels_explicit_only = test_labels[0]['explicit_only_output'].flatten()
test_labels_explicit_and_suggestive = test_labels[0]['explicit_and_suggestive_output'].flatten()


# %%
# Produce confusion matrix

# Convert the predicted classes from arrays to integers.
cfmx_cutoff = 0.6
pred_class_indices = np.where(predictions_explicit_only.flatten() < cfmx_cutoff, 0, 1)
true_class_indices = test_labels_explicit_only

# Get the confusion matrix using sklearn.
cfmx = skmetrics.confusion_matrix(y_true=pred_class_indices,  # True class for test-set.
                                  y_pred=true_class_indices)  # Predicted class.

draw_confusion_matrix(cfmx, HPARAMS['class_names'][0:2], (4, 3))


# %%
# PR Curves
pr_data_frames = []

class_true_labels = test_labels_explicit_only
class_predictions = predictions_explicit_only.flatten()
precision_1, recall_1, thresholds_1 = skmetrics.precision_recall_curve(class_true_labels, class_predictions)
average_precision = skmetrics.average_precision_score(class_true_labels, class_predictions)
pr_data_frames.append(pd.DataFrame({"Recall": recall_1, "Precision": precision_1,
                                    "Output": f'Explicit only (ap: {average_precision:.3f})'}))

class_true_labels = test_labels_explicit_and_suggestive
class_predictions = predictions_explicit_and_suggestive.flatten()
precision_2, recall_2, thresholds_2 = skmetrics.precision_recall_curve(class_true_labels, class_predictions)
average_precision = skmetrics.average_precision_score(class_true_labels, class_predictions)
pr_data_frames.append(pd.DataFrame({"Recall": recall_2, "Precision": precision_2,
                                    "Output": f'Explicit and suggestive (ap: {average_precision:.3f})'}))

pr_data_plot = pd.concat(pr_data_frames)

plt.figure(figsize=(15,14))
plt.grid()
plt.title(f'PR curves')
sns.set_style("dark", {"xtick.major.size": 16, "ytick.major.size": 16})
sns.lineplot(x="Recall", y="Precision", hue="Output", data=pr_data_plot)
plt.show()


# %%
# Explain images which were misclassified
explainer = GradCAM()

diff_class_indices = true_class_indices - pred_class_indices
false_indeces = np.argwhere(diff_class_indices != 0)
test_input_dataset_unbatched = test_input_dataset.unbatch()

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
# labels = (test_img_generator.class_indices)
# labels = dict((v, k) for k, v in labels.items())
# predictions_labeled = [labels[k] for k in pred_class_indices]

# pd.set_option('display.max_rows', None)
# cdf = pd.DataFrame({"Filename": test_img_generator.filenames[:len(test_classes)],
#                     "Prediction": predictions_labeled,
#                     "Benign Score": ["{0:.4f}".format(i[0]) for i in predictions],
#                     "Malign Score": ["{0:.4f}".format(i[1]) for i in predictions]})

# cdf.to_csv(output_path, encoding='utf-8', index=False)
