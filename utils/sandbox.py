# %%
import os
import matplotlib.pyplot as plt

from geacc.training import image_preprocessing

BATCH_SIZE = 10
IMG_SIZE = 299
VIS_DIR = "/Users/rustam/Development/purify/geacc-models/models/dataset-flowers/train"
CLASS_NUM = 3

# %%
def plot_images(images, cls_true, cls_pred=None, interpolation="spline16"):

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 4, figsize=(24, 24))

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# %%
_NUM_TRAIN_FILES = 1
_NUM_VALID_FILES = 1


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(
                data_dir, 'train-{:05d}-of-{:05d}'.format(i, _NUM_TRAIN_FILES))
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(
                data_dir, 'validation-{:05d}-of-{:05d}'.format(i, _NUM_VALID_FILES))
            for i in range(_NUM_VALID_FILES)]


train_input_dataset = image_preprocessing.input_fn(
    is_training=True,
    filenames=get_filenames(is_training=True, data_dir="/tmp"),
    batch_size=BATCH_SIZE,
    num_epochs=10,
    parse_record_fn=image_preprocessing.get_parse_record_fn(
        one_hot_encoding_class_num=3,
        use_keras_image_data_format=False),
    # datasets_num_private_threads=None,
    # dtype=tf.float32,
    # drop_remainder=False,
    # tf_data_experimental_slack=False,
    # training_dataset_cache=False,
)

features, labels = next(iter(train_input_dataset))
# plot_images(features[0:10],
#             tf.dtypes.cast(tf.reshape(labels, [10]), tf.int8), interpolation="lanczos")

# sys.exit()
