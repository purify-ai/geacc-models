# %%
import matplotlib.pyplot as plt

import os
import sys
import project_path

import tensorflow as tf
#import training.preprocess_crop

# print('\n'.join(sys.path))

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


# %%
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    # shear_range=0.5,
    # zoom_range = [0.8, 1], # only zoom-in
    rescale=1./255,
    # channel_shift_range=20
    # fill_mode='reflect'
    # preprocessing_function=preprocess_input
)

vis_img_generator = img_generator.flow_from_directory(
    VIS_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    interpolation='lanczos',
    shuffle=False)

# Wrap the keras generator
train_img_ds = tf.data.Dataset.from_generator(
    lambda: vis_img_generator,
    output_types=(tf.float32, tf.int8),
    output_shapes=([BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3],
                   [BATCH_SIZE]))

augmented_images, _ = vis_img_generator.next()
vis_classes = vis_img_generator.classes

class_names = list(vis_img_generator.class_indices.keys())
#print("""Class names: {}""".format(class_names))

steps = vis_img_generator.n
#print("""Steps on train: {}""".format(steps))

#plot_images(augmented_images[0:10], vis_classes[0:10], interpolation="lanczos")

features, labels = next(iter(train_img_ds))
#plot_images(features[0:10], labels[0:10], interpolation="lanczos")


# %%
# visualization of some images out of the image generator
#import Augmentor

BATCH_SIZE = 15
VIS_DIR = "/Users/rustam/Downloads/rips/"

#p = Augmentor.Pipeline(VIS_DIR)
#p.crop_by_size(probability=1, width=340, height=340, centre=False)
#p.crop_random(probability=1.0, percentage_area=0.875)
#p.resize(probability=1.0, width=IMG_SIZE, height=IMG_SIZE, resample_filter="LANCZOS")
# p.sample(20)

#aug_img_generator = p.keras_generator(batch_size=10, scaled=True)
#augmented_images, augmented_classes = next(aug_img_generator)

#plt.imshow(augmented_images[0].reshape(IMG_SIZE, IMG_SIZE, 3))
#plot_images(augmented_images, augmented_labels, interpolation="lanczos")
